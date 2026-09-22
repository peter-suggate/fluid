import { uniformPressureScratchShader } from "./uniform-scratch-arena";
import type { UniformTexturePages } from "./uniform-texture-pages";
import {uniformPressurePageExtent,uniformPressurePageWorkgroups,uniformPressurePagedShader,uniformPressurePageAddressWGSL} from "./uniform-pressure-pages";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { uniformPressureInPlaceSmootherWGSL, uniformPressureMultigridWGSL } from "./webgpu-uniform-pressure-multigrid.wgsl";

import {
  UNIFORM_CM11A_RECOVERY_BATCHES, UNIFORM_CM11A_RECOVERY_SWEEPS, UNIFORM_CM11A_PHI_PRESERVATION_LEVELS,
  UNIFORM_CM11A_COARSE_SWEEP_CAP, DEFAULT_UNIFORM_CM11A_SCHEDULE,
  type UniformCM11aSchedule,
} from "./pressure-policy";
export * from "./pressure-policy";
export * from "./pressure-plan";
import { UNIFORM_CM11A_COARSE_ROW_BYTES, UNIFORM_CM11A_COARSE_HEADER_BYTES } from "./uniform-coarse-solver.wgsl";
import { uniformAbOn } from "./uniform-ab-switch";
import { planUniformCM11aHierarchy, type UniformCM11aLevelSize } from "./pressure-plan";

/**
 * Profiling-only: append ` L<level>` to each multigrid compute-pass label.
 *
 * Every smoother/residual/restrict/prolong dispatch normally carries the same
 * per-kernel label at every level, so an external profiler (xctrace) and the
 * in-process pass-timestamp audit both collapse all levels into one bucket.
 * Reading the flag once at module scope keeps the encode path free of an
 * environment lookup, and leaving the flag unset reproduces the previous label
 * byte-for-byte.
 */
const BATCH_PASSES = uniformAbOn("batch");
const FUSE_VISITS = uniformAbOn("fusevisit");
/**
 * The recovery finish is always encoded and almost never runs (0 of 450 steps
 * across mini64, fig7 and the garden, 2026-09-21): 128 finest-level launches
 * whose every thread reads the gate and returns, ~18 us each at 128^3. "lagged"
 * launches them per row while the last observed step needed no recovery;
 * "force" does so unconditionally (measurement only).
 */
const ROW_SEGMENT = Math.max(1, Number((typeof process !== "undefined"
  ? process.env?.FLUID_UNIFORM_ROW_SEGMENT : undefined) ?? 8));
const ROW_SWEEP = !uniformAbOn("rowsweep") ? "off" : ((typeof process !== "undefined"
  ? process.env?.FLUID_UNIFORM_ROW_SWEEP : undefined) ?? "lagged");
const FUSED_VISIT_LANES = Math.max(1, Number((typeof process !== "undefined"
  ? process.env?.FLUID_UNIFORM_FUSED_VISIT_LANES : undefined) ?? 1024));
/**
 * Largest level (halo included) whose smoothing visit runs as one workgroup.
 * Measured on an M1 Max: 10^3 and below win (-2.6% mini64, -1.4% fig7 128^3);
 * at 18^3 a lane is ~35 serial updates per visit even 1024 wide and the fused
 * visit loses to twelve ~6 us launches on the GPU, though it still wins on CPU.
 */
const FUSED_VISIT_MAX_CELLS = Math.max(0, Number((typeof process !== "undefined"
  ? process.env?.FLUID_UNIFORM_FUSED_VISIT_CELLS : undefined) ?? 1000));
const MG_LEVEL_LABELS = typeof process !== "undefined"
  && process.env?.FLUID_UNIFORM_MG_LEVEL_LABELS === "1";

const ENTRY_POINTS = [
  "mgBuildFinestTopology", "mgBuildFinestRhs", "mgDownsampleTopology", "mgExtrapolatePhiOneCell",
  "mgBakeCoefficients",
  "mgResidual", "mgRestrictResidual", "mgProlongateAdd", "mgProlongateAssign",
  "mgDownsampleSubtract", "mgDownsampleMinimum", "mgSmoothColour", "mgSmoothColourInPlace", "mgSmoothRowInPlace", "mgSmoothVisitInPlace", "mgSaveAcceptedQuiet", "mgRestoreRejectedQuiet",
  "mgCopyPressure", "mgClearPressure", "mgClearMinimum",
  "mgPublishCycleDispatch", "mgShiftMinimum", "mgAddPressure", "mgSolveCoarsest", "mgMeasureFineResidual", "mgCheckCycleConvergence", "mgSaveAccepted", "mgRestoreRejected", "mgFinishSafety",
] as const;
type EntryPoint = typeof ENTRY_POINTS[number];

const ENTRY_BINDINGS: Readonly<Record<EntryPoint, readonly number[]>> = Object.freeze({
  mgBuildFinestTopology: [0, 6, 8], mgBuildFinestRhs: [0, 2, 4, 12],
  mgDownsampleTopology: [0, 5, 6, 7, 8], mgExtrapolatePhiOneCell: [0, 5, 6, 7],
  mgBakeCoefficients: [0, 5, 7, 15],
  mgResidual: [0, 1, 3, 10, 14], mgRestrictResidual: [0, 4, 9],
  mgProlongateAdd: [0, 1, 2, 9], mgProlongateAssign: [0, 1, 2],
  mgDownsampleSubtract: [0, 1, 11, 12], mgDownsampleMinimum: [0, 11, 12],
  mgSmoothColour: [0, 1, 2, 3, 11, 13, 14],
  mgSmoothColourInPlace: [0, 3, 11, 13, 14, 16],
  mgSmoothRowInPlace: [0, 3, 11, 13, 14, 16],
  mgSmoothVisitInPlace: [0, 3, 11, 13, 14, 16],
  mgCopyPressure: [0, 1, 2], mgClearPressure: [0, 2], mgClearMinimum: [0, 12],
  mgShiftMinimum: [0, 1, 11, 12], mgAddPressure: [0, 1, 2, 9],
  mgSolveCoarsest: [0, 1, 2, 3, 5, 7, 11, 13],
  mgMeasureFineResidual: [0, 1, 3, 11, 13, 14, 17],
  mgCheckCycleConvergence: [0, 17],
  mgPublishCycleDispatch: [0, 18],
  mgSaveAccepted: [0, 1, 2], mgRestoreRejected: [0, 2, 9],
  mgSaveAcceptedQuiet: [0, 1, 2], mgRestoreRejectedQuiet: [0, 2, 9], mgFinishSafety: [0],
});

/**
 * The same split, per entry point and computed once. The plan walk runs this
 * test thousands of times and rebuilding the sets and filtered lists inside it
 * was a measurable share of a re-plan.
 */
const SAMPLED_BINDINGS: readonly number[] = [1, 3, 5, 7, 9, 11, 14];
const WRITABLE_BINDINGS: readonly number[] = [2, 4, 6, 8, 10, 12, 15, 16];
const entryBindingsWhere = (keep: readonly number[]): Readonly<Record<EntryPoint, readonly number[]>> =>
  Object.freeze(Object.fromEntries((Object.entries(ENTRY_BINDINGS) as [EntryPoint, readonly number[]][])
    .map(([entry, bindings]) => [entry, Object.freeze(bindings.filter((b) => keep.includes(b)))]))) as
    Readonly<Record<EntryPoint, readonly number[]>>;
const ENTRY_SAMPLED = entryBindingsWhere(SAMPLED_BINDINGS);
const ENTRY_WRITABLE = entryBindingsWhere(WRITABLE_BINDINGS);

type TexturePair = readonly [GPUTexture, GPUTexture];
export interface UniformPressureMultigridLevel {
  /** Texture dimensions include one persistent solid/domain halo cell. */
  readonly dimensions: readonly [number, number, number];
  readonly pressure: TexturePair;
  readonly rhs: TexturePair;
  readonly phi: TexturePair;
  readonly volume: TexturePair;
  readonly residual: TexturePair;
  readonly minimum: TexturePair;
  /** Immutable (+x,+y,+z) pressure coefficients and liquid flag for this solve. */
  readonly coefficients: GPUTexture;
}

/**
 * The cycle-schedule group a planned dispatch belongs to, so a trace seam can
 * split the fixed CM11a schedule into its meaningful sections without timing
 * ~2000 hierarchy passes individually: the topology/RHS pyramid setup, the
 * three Full-Cycles, the four V-Cycles, and the parity copy + fine-residual
 * measure that close the solve.
 */
export type UniformCM11aPlanStage = "setup" | "full-cycle" | "v-cycle" | "finish";

interface PlannedDispatch {
  readonly pipeline: GPUComputePipeline;
  readonly group: GPUBindGroup;
  readonly entryPoint: EntryPoint;
  readonly stage: UniformCM11aPlanStage;
  readonly workgroups: readonly [number, number, number];
  readonly activeLevel: number;
  readonly residualCheckpoint: boolean;
  readonly cycleGate: number;
  /**
   * The same update launched one thread per row instead of one per cell. It is
   * several times slower when the gate is open and costs a launch floor rather
   * than a lattice of early-outs when it is closed; values are identical.
   */
  quiet?: { readonly pipeline: GPUComputePipeline; readonly workgroups: readonly [number, number, number] };
  readonly coarsestCapture?: {
    readonly invocation: number;
    readonly dimensions: readonly [number, number, number];
    readonly pressure: GPUTexture; readonly rhs: GPUTexture; readonly minimum: GPUTexture;
    readonly phi: GPUTexture; readonly topology: GPUTexture;
  };
}

export interface UniformCM11aCoarsestCapture {
  readonly dimensions: readonly [number, number, number];
  readonly pressure: readonly number[]; readonly rhs: readonly number[];
  readonly minimum: readonly number[]; readonly phi: readonly number[];
  /** Flat rgba tuples: cell V followed by positive x/y/z dual volumes. */
  readonly topology: readonly number[];
}

interface CoarsestCaptureBuffers {
  readonly invocation: number;
  readonly dimensions: readonly [number, number, number];
  readonly byteLength: number;
  readonly bytesPerRow: number;
  readonly pressure: GPUBuffer; readonly rhs: GPUBuffer; readonly minimum: GPUBuffer;
  readonly phi: GPUBuffer; readonly topology: GPUBuffer;
}

interface GroupResources {
  pressureIn: GPUTexture; pressureOut: GPUTexture;
  rhsIn: GPUTexture; rhsOut: GPUTexture;
  phiIn: GPUTexture; phiOut: GPUTexture;
  volumeIn: GPUTexture; volumeOut: GPUTexture;
  residualIn: GPUTexture; residualOut: GPUTexture;
  minimumIn: GPUTexture; minimumOut: GPUTexture;
  coefficientsIn: GPUTexture; coefficientsOut: GPUTexture;
  /** The in-place smoother's one pressure texture; unset everywhere else. */
  pressureRW?: GPUTexture;
}

/**
 * Uniform implementation of CM11a Algorithms 1--3 with paged or dense backing. The shader source
 * passed to initialize must be the authoritative uniform shader followed by
 * {@link uniformPressureMultigridWGSL}; this lets mgBuildFinest call exactly
 * the same solid, free-surface, divergence, and volume-correction helpers as
 * projection instead of maintaining a second discretization.
 */
/**
 * Everything a CM11a instance compiles that does not depend on its dimensions.
 *
 * Bind-group layouts are built from a fixed binding table and the pipelines
 * from a fixed entry-point list, so two instances of different capacity share
 * both. That is what makes a re-plan synchronous and cheap: a new instance is
 * textures, param buffers, bind groups and a plan, with nothing to compile and
 * nothing to await.
 */
export interface UniformPressureMultigridPrograms {
  readonly pagedStorage: boolean;
  readonly module: GPUShaderModule;
  readonly groupLayouts: Readonly<Record<string, GPUBindGroupLayout>>;
  readonly pipelines: Readonly<Record<string, GPUComputePipeline>>;
}

export class WebGPUUniformPressureMultigrid {
  readonly levels: readonly UniformPressureMultigridLevel[];
  get shaderFragment():string {
    if(this.pagedStorage) return uniformPressurePagedShader(uniformPressureMultigridWGSL,this.logicalPageDispatch);
    // Without an active-region dispatch buffer there is no window and no level
    // origin: the host seeds every level record at zero with no clip extent and
    // the GPU finalize that would move them never runs. Every thread of every
    // level kernel was still reading five storage words to add that zero.
    const smoother=this.inPlaceCapable?uniformPressureInPlaceSmootherWGSL:"";
    if(!this.activeDispatch&&uniformAbOn("mgstaticid")){
      const dynamic="fn mgActiveId(gid:vec3u)->vec3i{";
      if(!uniformPressureMultigridWGSL.includes(dynamic)) throw new Error("mgActiveId specialisation lost its anchor");
      const source=uniformPressureMultigridWGSL.replace(dynamic,`${dynamic}\n  if(true){return vec3i(gid);}`)+smoother;
      return this.scratchFields?uniformPressureScratchShader(source):source;
    }
    return this.scratchFields?uniformPressureScratchShader(uniformPressureMultigridWGSL+smoother):uniformPressureMultigridWGSL+smoother;
  }
  private readonly logicalDimensions = new Map<GPUTexture,readonly [number,number,number]>();
  private pressurePublication?: GPUTexture;
  private pressurePublicationPipeline?: GPUComputePipeline;
  private pressurePublicationGroup?: GPUBindGroup;
  private pageCapturePipeline?: GPUComputePipeline;
  private readonly pageCaptureBuffers:GPUBuffer[]=[];
  private readonly pageCaptureOperations=new Map<GPUTexture,{group:GPUBindGroup;staging:GPUBuffer;target:GPUBuffer}>();
  readonly diagnostics: GPUBuffer;
  private readonly toleranceBuffer: GPUBuffer;
  private readonly cycleDispatch: GPUBuffer;
  readonly allocatedBytes: number;
  /** CM11a Algorithm 3 p_tmp; no V-cycle scratch dispatch may alias it. */
  private readonly fullCycleBackup: GPUTexture;
  private readonly acceptedPressure: GPUTexture;
  private readonly spacing: readonly [number, number, number];
  /** Finest physical lattice, the reference every level's spacing scales from. */
  private readonly finestSize!: UniformCM11aLevelSize;
  private readonly groupLayouts: Readonly<Record<EntryPoint, GPUBindGroupLayout>>;
  private readonly ownedParams: GPUBuffer[] = [];
  private readonly ownedGroups: GPUBindGroup[] = [];
  /**
   * The plan is thousands of dispatches over a few dozen textures, and the
   * same (entry point, resources, parameters) triple recurs on every cycle of
   * every level. Building each of the three afresh per dispatch is what made a
   * re-plan a visible hitch: on the whole 64x512x64 domain, 59k texture views,
   * 4.2k parameter buffers and 4.2k bind groups cost 143 ms of the 178.
   * Caching them is pure de-duplication -- a bind group is immutable, and two
   * dispatches naming the same resources need only one -- so the encoded
   * command stream is unchanged.
   */
  private readonly viewCache = new Map<GPUTexture, GPUTextureView>();
  private readonly paramCache = new Map<string, GPUBuffer>();
  private readonly groupCache = new Map<string, GPUBindGroup>();
  /** Stable per-texture ids, so a bind group's resources have a cache key. */
  private readonly textureIds = new Map<GPUTexture, number>();
  private pipelines?: Readonly<Record<EntryPoint, GPUComputePipeline>>;
  private shaderModule?: GPUShaderModule;
  private plan?: readonly PlannedDispatch[];
  /** A deferred build in progress; see `advancePlan`. */
  private planSteps?: Generator<void, PlannedDispatch[], void>;
  /**
   * Plan index after each complete cycle, `[setupEnd, afterCycle1, ...]`, so
   * entry `k` is where a budget of `k` cycles stops encoding. Every cycle ends
   * on its checkpoint, which canonicalizes finest pressure into parity A, and
   * the finish dispatches are bound to parity A — so truncating here is the
   * only place the encoded prefix can be cut without rebuilding bind groups.
   */
  private cycleBoundaries?: readonly number[];
  /** First plan index of the always-encoded finish section. */
  private finishStart = 0;
  private activeResidualTolerance = 0;
  private coarsestCaptureBuffers?: CoarsestCaptureBuffers;
  private windowLevelGroups?: readonly (readonly [number, number, number])[];
  private windowLattice = false;
  private isDestroyed = false;
  private readonly inPlaceCapable: boolean;
  /** Lanes of the fused smoothing visit: as wide as one workgroup may be. */
  private readonly visitLanes: number;
  private inPlaceSmoothing: boolean;

  constructor(private readonly device: GPUDevice,
    dimensions: readonly [number, number, number],
    spacing: readonly [number, number, number],
    private readonly schedule: UniformCM11aSchedule = DEFAULT_UNIFORM_CM11A_SCHEDULE,
    private readonly activeDispatch?: GPUBuffer,
    programs?: UniformPressureMultigridPrograms,
    /**
     * Allocate the lattice but leave the plan to `advancePlan`. A prewarm
     * builds the capacity the window is about to need across several frames,
     * so the step that switches only swaps an instance that is already ready.
     */
    deferPlan = false,
    referenceDimension: 2 | 3 = 3,
    private readonly gpuCycleDispatch = false,
    private readonly pagedStorage = false,
    /** QA attribution: atlas storage with exact logical launches. */
    private readonly logicalPageDispatch = false,
    /**
     * Smooth each colour in place over half the lattice. The caller vouches
     * that the scene has no depth symmetry; the full-lattice and dense-storage
     * conditions are checked here.
     */
    inPlaceSmoothing = false, private readonly scratchFields?: UniformTexturePages) {
    this.inPlaceCapable = inPlaceSmoothing && !activeDispatch && !pagedStorage && programs === undefined;
    this.inPlaceSmoothing = this.inPlaceCapable;
    this.visitLanes = Math.min(FUSED_VISIT_LANES, device.limits.maxComputeInvocationsPerWorkgroup,
      device.limits.maxComputeWorkgroupSizeX);
    const hierarchy = planUniformCM11aHierarchy(
      dimensions as readonly [number, number, number], referenceDimension);
    if (hierarchy.rejection) throw new RangeError(hierarchy.rejection);
    if (!spacing.every((value) => Number.isFinite(value) && value > 0)) {
      throw new RangeError("CM11a grid spacing must be positive and finite");
    }
    const coarseScratchBytes = hierarchy.coarsestCells * UNIFORM_CM11A_COARSE_ROW_BYTES;
    const stateBytes = UNIFORM_CM11A_COARSE_HEADER_BYTES + coarseScratchBytes;
    if (stateBytes > Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize)) {
      throw new RangeError(`CM11a coarse grid ${hierarchy.levelDimensions.at(-1)!.join("x")} needs ${coarseScratchBytes} scratch bytes, exceeding device buffer limits`);
    }
    this.spacing = spacing;
    const { levelCount } = hierarchy;
    this.finestSize = hierarchy.levelDimensions[0]!;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    let allocatedBytes = 120;
    const texture = (label: string, format: GPUTextureFormat,
      size: readonly [number, number, number]) => {
      const extent=this.pagedStorage?uniformPressurePageExtent(size):size;
      const result = (this.scratchFields ?? device).createTexture({ label, size: [...extent], dimension: "3d", format, usage });
      this.logicalDimensions.set(result,size);
      allocatedBytes += extent[0] * extent[1] * extent[2] * (format === "rgba32float" ? 16 : 4);
      return result;
    };
    const levels: UniformPressureMultigridLevel[] = [];
    for (let index = 0; index < levelCount; index += 1) {
      const physicalSize = hierarchy.levelDimensions[index]!;
      const size: readonly [number, number, number] = [physicalSize[0] + 2, physicalSize[1] + 2, physicalSize[2] + 2];
      const pair = (field: string, format: GPUTextureFormat = "r32float"): TexturePair => [
        texture(`Uniform CM11a L${index} ${field} A`, format, size),
        texture(`Uniform CM11a L${index} ${field} B`, format, size),
      ];
      // These fields never ping-pong: the plan always addresses slot zero.
      // Keep the pair-shaped resource interface without allocating an unused
      // second volume or residual field at every level.
      const single = (field: string, format: GPUTextureFormat = "r32float"): TexturePair => {
        const value = texture(`Uniform CM11a L${index} ${field} A`, format, size);
        return [value, value];
      };
      levels.push(Object.freeze({ dimensions: size, pressure: pair("pressure"), rhs: pair("rhs"),
        phi: pair("phi"), volume: single("V", "rgba32float"), residual: single("residual"),
        minimum: pair("p-min"), coefficients: texture(`Uniform CM11a L${index} coefficients`, "rgba32float", size) }));
    }
    this.levels = Object.freeze(levels);
    // Three banks: immutable launch geometry, normal cycles, recovery cycles.
    // The final record in each bank is the single-workgroup coarse solve.
    const records = levels.length + 1;
    const cycleWords = new Uint32Array(records * 9);
    levels.forEach((level, index) => cycleWords.set(this.pagedStorage && !this.logicalPageDispatch?uniformPressurePageWorkgroups(level.dimensions):level.dimensions.map(n => Math.ceil(n / 4)), index * 3));
    cycleWords.set([1, 1, 1], levels.length * 3);
    this.cycleDispatch = device.createBuffer({label: "Uniform GPU pressure cycle dispatch", size: cycleWords.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(this.cycleDispatch, 0, cycleWords);
    allocatedBytes += cycleWords.byteLength;
    if(this.pagedStorage){
      const size=levels[0]!.dimensions;
      this.pressurePublication=device.createTexture({label:"Uniform pressure presentation adapter",size:[...size],dimension:"3d",format:"r32float",usage});
      allocatedBytes+=size[0]*size[1]*size[2]*4;
    }
    this.fullCycleBackup = texture("Uniform CM11a Full-Cycle p_tmp", "r32float", levels[0]!.dimensions);
    this.acceptedPressure = texture("Uniform CM11a accepted pressure", "r32float", levels[0]!.dimensions);
    this.diagnostics = device.createBuffer({ label: "Uniform CM11a convergence status", size: stateBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.toleranceBuffer = device.createBuffer({ label: "Pressure residual tolerance", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.setResidualTolerance(schedule.residualTolerance ?? 10);
    allocatedBytes += stateBytes - 104;
    this.allocatedBytes = allocatedBytes;
    const textureBinding = { sampleType: "unfilterable-float", viewDimension: "3d" } as const;
    const scalarStorage = { access: "write-only", format: "r32float", viewDimension: "3d" } as const;
    const allEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 18, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ...[1, 3, 5, 7, 9, 11].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, texture: textureBinding })),
      ...[2, 4, 6, 10, 12].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, storageTexture: scalarStorage })),
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 14, visibility: GPUShaderStage.COMPUTE, texture: textureBinding },
      { binding: 15, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "read-write", format: "r32float", viewDimension: "3d" } },
    ];
    this.groupLayouts = (programs?.groupLayouts as Record<EntryPoint, GPUBindGroupLayout> | undefined)
      ?? Object.freeze(Object.fromEntries(ENTRY_POINTS.map((entryPoint) => [entryPoint,
        device.createBindGroupLayout({ label: `Uniform CM11a hierarchy layout - ${entryPoint}`,
          entries: allEntries.filter(({ binding }) => (binding === 13 || ENTRY_BINDINGS[entryPoint].includes(binding))) }),
      ])) as Record<EntryPoint, GPUBindGroupLayout>);
    // Given another instance's compiled programs this one is ready here, with
    // no await anywhere: a re-plan happens inside a single step.
    if (programs) {
      if(programs.pagedStorage!==this.pagedStorage)throw new Error("Cannot reuse pressure programs across field layouts");
      this.shaderModule = programs.module;
      this.pipelines = programs.pipelines as Readonly<Record<EntryPoint, GPUComputePipeline>>;
      if (deferPlan) this.planSteps = this.buildPlanSteps();
      else this.plan = Object.freeze(this.buildPlan());
    }
  }

  /**
   * The compiled, dimension-independent half of this instance, for handing to
   * a sibling planned on a different capacity. Undefined until initialized.
   */
  get programs(): UniformPressureMultigridPrograms | undefined {
    // Paged instances own dimension-specific publication resources; only the
    // legacy window backend supports synchronous program reuse.
    if (this.pagedStorage || !this.shaderModule || !this.pipelines) return undefined;
    return { module: this.shaderModule, groupLayouts: this.groupLayouts, pipelines: this.pipelines, pagedStorage: this.pagedStorage };
  }

  setResidualTolerance(value: number): void {
    const tolerance = Number.isFinite(value) ? Math.max(0, value) : 0;
    this.activeResidualTolerance = tolerance;
    this.device.queue.writeBuffer(this.toleranceBuffer, 0, new Float32Array([tolerance, 0, 0, 0]));
  }

  /**
   * The tolerance the GPU gate is currently comparing against. Zero disables
   * the early exit entirely, which is also the only state in which a cycle
   * count carries no information about how many cycles the solve needed.
   */
  get residualTolerance(): number { return this.activeResidualTolerance; }

  get pressureTexture(): GPUTexture { return this.pressurePublication??this.levels[0]!.pressure[0]; }

  /**
   * Per-level group counts the host chose for this step, or undefined to keep
   * taking them from the GPU's indirect records.
   *
   * The level ORIGIN still comes from `activeRegion` inside the shader, so
   * this only replaces the launch size -- and only ever with a count that is
   * at most the dense one the plan already carries. The plan itself is
   * untouched: the same dispatches are encoded in the same order, so the
   * encoded pass count cannot move with the window.
   */
  setWindowLevelGroups(groups?: readonly (readonly [number, number, number])[]): void {
    this.windowLevelGroups = groups;
  }

  /**
   * Whether this instance IS the window.
   *
   * A lattice planned on the window covers exactly the cells it owns, so every
   * dispatch is the plan's own dense count, direct, with no origin to add and
   * no level record to consult. The encoded pass count is the same either way.
   */
  setWindowLattice(active: boolean): void { this.windowLattice = active; }

  /** True once `destroy` has run; an evicted instance must not be adopted. */
  get destroyed(): boolean { return this.isDestroyed; }

  /**
   * The in-place smoother's colouring must separate all six neighbours, which
   * it does not under depth symmetry. A live edit that introduces it falls back
   * to the ping-pong smoother; the plan is rebuilt once, on that edit.
   */
  setDepthSymmetry(symmetric: boolean): void {
    const next = this.inPlaceCapable && !symmetric;
    if (next === this.inPlaceSmoothing) return;
    this.inPlaceSmoothing = next;
    if (this.pipelines) this.plan = Object.freeze(this.buildPlan());
  }

  /** Physical (halo-free) dimensions used to seed the shared active ABI. */
  get levelPhysicalDimensions(): readonly UniformCM11aLevelSize[] {
    return this.levels.map((level) => [level.dimensions[0] - 2,
      level.dimensions[1] - 2, level.dimensions[2] - 2] as const);
  }

  async initialize(input: { readonly uniformBindGroupLayout: GPUBindGroupLayout;
    readonly shaderSource: string; readonly signal?: AbortSignal }): Promise<void> {
    this.assertLive(); if (this.pipelines) return;
    const compiler = gpuCompilationManagerFor(this.device);
    const shaderModule = compiler.createShaderModule({ label: "Uniform CM11a pressure hierarchy",
      code: input.shaderSource });
    this.shaderModule = shaderModule;
    const emptyUniformLayout = this.device.createBindGroupLayout({entries: []});
    const entries = await Promise.all(ENTRY_POINTS.filter((entryPoint) =>
      !/InPlace$|Quiet$/.test(entryPoint) || this.inPlaceCapable).map(async (entryPoint) => [entryPoint,
      await compiler.compileComputePipeline({ label: `Uniform CM11a - ${entryPoint}`,
        layout: this.device.createPipelineLayout({ label: `Uniform CM11a layout - ${entryPoint}`,
          bindGroupLayouts: [entryPoint === "mgPublishCycleDispatch" ? emptyUniformLayout : input.uniformBindGroupLayout, this.groupLayouts[entryPoint]] }),
        compute: { module: shaderModule, entryPoint,
          ...(entryPoint === "mgSmoothVisitInPlace" ? { constants: { MG_VISIT_LANES: this.visitLanes } } : {}),
          ...(entryPoint === "mgSmoothRowInPlace" || /Quiet$/.test(entryPoint)
            ? { constants: { MG_ROW_SEGMENT: ROW_SEGMENT } } : {}) } },
        { priority: "visible", signal: input.signal })] as const));
    this.pipelines = Object.freeze(Object.fromEntries(entries) as Record<EntryPoint, GPUComputePipeline>);
    this.plan = Object.freeze(this.buildPlan());
    if(this.pagedStorage){
      const dims=this.levels[0]!.dimensions;
      const module=compiler.createShaderModule({label:"Uniform paged pressure publication",code:`
        @group(0) @binding(0) var input:texture_3d<f32>;
        @group(0) @binding(1) var output:texture_storage_3d<r32float,write>;
        ${uniformPressurePageAddressWGSL}
        @compute @workgroup_size(4,4,4) fn publish(@builtin(global_invocation_id)g:vec3u){
          let d=vec3u(${dims.join("u,")}u);if(any(g>=d)){return;}
          textureStore(output,vec3i(g),textureLoad(input,mgPageAddress(vec3i(g),d,textureDimensions(input)),0));
        }`});
      this.pressurePublicationPipeline=await compiler.compileComputePipeline({label:"Uniform pressure page publication",layout:"auto",compute:{module,entryPoint:"publish"}},{priority:"visible",signal:input.signal});
      this.pressurePublicationGroup=this.device.createBindGroup({layout:this.pressurePublicationPipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:this.levels[0]!.pressure[0].createView()},
        {binding:1,resource:this.pressurePublication!.createView()},
      ]});
      const captureModule=compiler.createShaderModule({label:"Uniform pressure page capture",code:`
        struct Capture {dims:vec4u,packing:vec4u}
        @group(0) @binding(0) var input:texture_3d<f32>;
        @group(0) @binding(1) var<storage,read_write> output:array<f32>;
        @group(0) @binding(2) var<uniform> capture:Capture;
        ${uniformPressurePageAddressWGSL}
        @compute @workgroup_size(64) fn captureField(@builtin(global_invocation_id)g:vec3u){
          let d=capture.dims.xyz;let i=g.x;if(i>=d.x*d.y*d.z){return;}
          let q=vec3u(i%d.x,(i/d.x)%d.y,i/(d.x*d.y));
          let v=textureLoad(input,mgPageAddress(vec3i(q),d,textureDimensions(input)),0);
          let offset=(q.z*d.y+q.y)*capture.packing.x+q.x*capture.dims.w;
          for(var component=0u;component<capture.dims.w;component++){output[offset+component]=v[component];}
        }`});
      this.pageCapturePipeline=await compiler.compileComputePipeline({label:"Uniform pressure page capture",layout:"auto",compute:{module:captureModule,entryPoint:"captureField"}},{priority:"background",signal:input.signal});
    }
  }

  /**
   * Encode the configured Full-Cycle/V-Cycle schedule, or its first
   * `cycleBudget` cycles.
   *
   * Truncation drops the tail cycles from the command stream entirely: fewer
   * `beginComputePass` calls, fewer dispatches, less CPU encode. The setup
   * pyramid and the finish section are always encoded. The GPU-side
   * `mgSkipCycle` gate is untouched and remains the inner stop, so a step that
   * converges inside its budget still exits early.
   */
  encode(
    encoder: GPUCommandEncoder,
    uniformGroup: GPUBindGroup,
    boundary?: (stage: UniformCM11aPlanStage) => void,
    cycleBudget?: number,
    /** The last observed step entered recovery; launch its sweeps per cell. */
    recoveryExpected = true,
  ): void {
    this.assertLive(); if (!this.plan) throw new Error("Uniform CM11a hierarchy is not initialized");
    encoder.clearBuffer(this.diagnostics, 0, 104);
    const prefixEnd = this.cycleBoundaries?.[this.clampCycleBudget(cycleBudget)] ?? this.plan.length;
    let openStage: UniformCM11aPlanStage | undefined;
    // A WebGPU usage scope is one dispatch, not one compute pass, so a storage
    // write here and a sampled read of the same texture in the next dispatch
    // may share a pass (2000-dispatch ping-pong probe, validation on, 2026-09-21).
    // Dawn already merges adjacent passes into one Metal encoder, so this saves
    // no GPU time; it removes the per-pass begin/end and the rebinding of the
    // 29-entry group 0 on the CPU. Profiling lanes keep one labelled pass per
    // dispatch so external traces stay attributable per kernel.
    const batch = BATCH_PASSES && !MG_LEVEL_LABELS;
    let shared: GPUComputePassEncoder | undefined, sharedHasGroup0 = false;
    const closeShared = () => { shared?.end(); shared = undefined; sharedHasGroup0 = false; };
    for (let index = 0; index < this.plan.length; index += 1) {
      const dispatch = this.plan[index]!;
      if (openStage !== undefined && dispatch.stage !== openStage) { closeShared(); boundary?.(openStage); }
      openStage = dispatch.stage;
      // A truncated cycle encodes nothing at all. Its stage seam is still
      // reported above, in order, so the advance's phase partition keeps all
      // four sections and a section with no passes reads as zero-length
      // instead of vanishing from the trace.
      if (index >= prefixEnd && index < this.finishStart) continue;
      if (dispatch.residualCheckpoint) {
        closeShared();
        encoder.clearBuffer(this.diagnostics, 60, 4);
      }
      const pass = shared ?? encoder.beginComputePass({ label: batch ? "Uniform CM11a pressure cycle" : MG_LEVEL_LABELS
        ? `Uniform CM11a ${dispatch.entryPoint} L${dispatch.activeLevel}`
        : `Uniform CM11a ${dispatch.entryPoint}` });
      if (batch) shared = pass;
      const quiet = (ROW_SWEEP === "force" || !recoveryExpected) && !this.gpuCycleDispatch
        && !this.windowLattice && !this.activeDispatch ? dispatch.quiet : undefined;
      pass.setPipeline(quiet?.pipeline ?? dispatch.pipeline); pass.setBindGroup(1, dispatch.group);
      if (dispatch.entryPoint !== "mgPublishCycleDispatch" && !sharedHasGroup0) {
        pass.setBindGroup(0, uniformGroup); sharedHasGroup0 = batch;
      }
      if (this.gpuCycleDispatch && dispatch.cycleGate !== 0) {
        const record = dispatch.entryPoint === "mgSolveCoarsest" ? this.levels.length : dispatch.activeLevel;
        pass.dispatchWorkgroupsIndirect(this.cycleDispatch,
          (dispatch.cycleGate * (this.levels.length + 1) + record) * 12);
      } else if (this.windowLattice) {
        pass.dispatchWorkgroups(...dispatch.workgroups);
      } else if (this.activeDispatch && dispatch.entryPoint !== "mgSolveCoarsest" && dispatch.entryPoint !== "mgCheckCycleConvergence" && dispatch.entryPoint !== "mgFinishSafety") {
        const chosen = this.windowLevelGroups?.[dispatch.activeLevel];
        if (chosen) {
          pass.dispatchWorkgroups(Math.min(chosen[0], dispatch.workgroups[0]),
            Math.min(chosen[1], dispatch.workgroups[1]), Math.min(chosen[2], dispatch.workgroups[2]));
        } else {
          const indirectOffset = (16 + dispatch.activeLevel * 10 + 3) * 4;
          pass.dispatchWorkgroupsIndirect(this.activeDispatch, indirectOffset);
        }
      } else {
        pass.dispatchWorkgroups(...(quiet?.workgroups ?? dispatch.workgroups));
      }
      if (!batch) pass.end();
      // This kernel's layout carries no group 0; rebind for whatever follows.
      else if (dispatch.entryPoint === "mgPublishCycleDispatch") sharedHasGroup0 = false;
      if (dispatch.coarsestCapture && this.coarsestCaptureBuffers
        && dispatch.coarsestCapture.invocation === this.coarsestCaptureBuffers.invocation) {
        closeShared();
        const capture = dispatch.coarsestCapture, buffers = this.coarsestCaptureBuffers;
        const destination = (buffer: GPUBuffer) => ({ buffer, bytesPerRow: buffers.bytesPerRow,
          rowsPerImage: capture.dimensions[1] });
        if(this.pagedStorage){
          for(const field of [capture.pressure,capture.rhs,capture.minimum,capture.phi,capture.topology]){
            const operation=this.pageCaptureOperations.get(field)!;
            const pass=encoder.beginComputePass({label:"Capture logical pressure page field"});
            pass.setPipeline(this.pageCapturePipeline!);pass.setBindGroup(0,operation.group);
            pass.dispatchWorkgroups(Math.ceil(capture.dimensions.reduce((n,d)=>n*d,1)/64));pass.end();
            encoder.copyBufferToBuffer(operation.staging,0,operation.target,0,buffers.byteLength);
          }
        }else{
          for(const name of ["pressure","rhs","minimum","phi","topology"] as const){
            const field=capture[name];
            const texture=this.scratchFields?.snapshotTexture(field) ?? field;
            this.scratchFields?.encodeSnapshot(encoder,field);
            encoder.copyTextureToBuffer({texture},destination(buffers[name]),capture.dimensions);
          }
        }
      }
    }
    closeShared();
    if(this.pressurePublicationPipeline){
      const pass=encoder.beginComputePass({label:"Publish paged pressure for projection"});
      pass.setPipeline(this.pressurePublicationPipeline);pass.setBindGroup(0,this.pressurePublicationGroup!);
      pass.dispatchWorkgroups(...this.levels[0]!.dimensions.map(n=>Math.ceil(n/4)) as [number,number,number]);pass.end();
    }
    if (openStage !== undefined) boundary?.(openStage);
  }

  get levelCount(): number { return this.levels.length; }

  /** Cycles the built plan carries: Full-Cycles plus V-Cycles. */
  get cycleCount(): number { return Math.max(0, (this.cycleBoundaries?.length ?? 1) - 1); }

  /** Compute passes the whole plan encodes. Undefined until `initialize`. */
  get planPassCount(): number | undefined { return this.plan ? this.plan.length+Number(this.pagedStorage) : undefined; }

  /** Compute passes a given cycle budget encodes, setup and finish included. */
  encodedPassCount(cycleBudget?: number): number | undefined {
    if (!this.plan || !this.cycleBoundaries) return undefined;
    return this.cycleBoundaries[this.clampCycleBudget(cycleBudget)]!
      + (this.plan.length - this.finishStart) + Number(this.pagedStorage);
  }

  private clampCycleBudget(cycleBudget?: number): number {
    const total = this.cycleCount;
    if (cycleBudget === undefined || !Number.isFinite(cycleBudget)) return total;
    return Math.min(total, Math.max(0, Math.floor(cycleBudget)));
  }

  /** Compute passes per advance in each cycle-schedule group, from the plan
   * actually built for this grid. Undefined until `initialize` resolves. */
  get planStageCounts(): Readonly<Record<UniformCM11aPlanStage, number>> | undefined {
    if (!this.plan) return undefined;
    const counts: Record<UniformCM11aPlanStage, number> = {
      setup: 0, "full-cycle": 0, "v-cycle": 0, finish: 0,
    };
    for (const dispatch of this.plan) counts[dispatch.stage] += 1;
    counts.finish+=Number(this.pagedStorage);
    return counts;
  }

  /** Opt-in diagnostic capture of one coarsest solve invocation. It does not alter the solve. */
  enableCoarsestCapture(invocation = 1): void {
    this.assertLive(); if (this.coarsestCaptureBuffers) return;
    if (!Number.isSafeInteger(invocation) || invocation < 1) {
      throw new RangeError("CM11a coarsest capture invocation must be a positive integer");
    }
    const dimensions = this.levels.at(-1)!.dimensions;
    const bytesPerRow = Math.ceil(dimensions[0] * 16 / 256) * 256;
    const byteLength = bytesPerRow * dimensions[1] * dimensions[2];
    const buffer = (label: string) => this.device.createBuffer({ label, size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.coarsestCaptureBuffers = { invocation, dimensions, byteLength, bytesPerRow,
      pressure: buffer("Uniform CM11a capture pressure"), rhs: buffer("Uniform CM11a capture rhs"),
      minimum: buffer("Uniform CM11a capture p-min"), phi: buffer("Uniform CM11a capture phi"),
      topology: buffer("Uniform CM11a capture topology") };
    if(this.pagedStorage){
      const capture=this.plan?.find(d=>d.coarsestCapture?.invocation===invocation)?.coarsestCapture;
      if(!capture||!this.pageCapturePipeline)throw new Error("Paged pressure capture requires an initialized invocation");
      const target=this.coarsestCaptureBuffers;
      for(const name of ["pressure","rhs","minimum","phi","topology"] as const){
        const staging=this.device.createBuffer({label:`Page capture ${name}`,size:byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
        const params=this.device.createBuffer({label:"Page capture geometry",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
        this.device.queue.writeBuffer(params,0,new Uint32Array([...dimensions,name==="topology"?4:1,bytesPerRow/4,0,0,0]));
        const group=this.device.createBindGroup({layout:this.pageCapturePipeline.getBindGroupLayout(0),entries:[
          {binding:0,resource:capture[name].createView()},{binding:1,resource:{buffer:staging}},{binding:2,resource:{buffer:params}},
        ]});
        this.pageCaptureBuffers.push(staging,params);
        this.pageCaptureOperations.set(capture[name],{group,staging,target:target[name]});
      }
    }
  }

  async readCoarsestCapture(): Promise<UniformCM11aCoarsestCapture | undefined> {
    const capture = this.coarsestCaptureBuffers; if (!capture) return undefined;
    await this.device.queue.onSubmittedWorkDone();
    const buffers = [capture.pressure, capture.rhs, capture.minimum, capture.phi, capture.topology];
    await Promise.all(buffers.map((buffer) => buffer.mapAsync(GPUMapMode.READ)));
    try {
      const [dx, dy, dz] = capture.dimensions;
      const unpack = (buffer: GPUBuffer, components: number) => {
        const bytes = new Uint8Array(buffer.getMappedRange()); const values: number[] = [];
        for (let z = 0; z < dz; z += 1) for (let y = 0; y < dy; y += 1) {
          const row = new Float32Array(bytes.buffer, bytes.byteOffset + capture.bytesPerRow * (y + dy * z), capture.bytesPerRow / 4);
          for (let x = 0; x < dx; x += 1) for (let c = 0; c < components; c += 1) {
            values.push(row[components * x + c]!);
          }
        }
        return values;
      };
      return Object.freeze({ dimensions: capture.dimensions,
        pressure: unpack(capture.pressure, 1), rhs: unpack(capture.rhs, 1),
        minimum: unpack(capture.minimum, 1), phi: unpack(capture.phi, 1),
        topology: unpack(capture.topology, 4) });
    } finally { for (const buffer of buffers) buffer.unmap(); }
  }

  /** One view per texture; the plan asks for the same handful thousands of times. */
  private viewOf(texture: GPUTexture): GPUTextureView {
    let view = this.viewCache.get(texture);
    if (!view) { view = texture.createView(); this.viewCache.set(texture, view); }
    return view;
  }

  private textureId(texture: GPUTexture): number {
    let id = this.textureIds.get(texture);
    if (id === undefined) { id = this.textureIds.size; this.textureIds.set(texture, id); }
    return id;
  }

  /**
   * Walk the schedule, yielding at every cycle boundary.
   *
   * A capacity change rebuilds this, and on the whole domain that is thousands
   * of dispatches: too much for one frame. Every cycle is a natural pause --
   * the walk's only state between them is the ping-pong parities and the
   * boundary list -- so a prewarm can spend a millisecond or two a frame here
   * and have the instance ready before the window needs it.
   */
  private *buildPlanSteps(): Generator<void, PlannedDispatch[], void> {
    const result: PlannedDispatch[] = [];
    // The schedule group each emit lands in; reassigned as the plan walks its
    // fixed sections so every dispatch self-reports where it sits.
    let planStage: UniformCM11aPlanStage = "setup";
    let recovering = false;
    const p = new Array(this.levels.length).fill(0);
    const phi = new Array(this.levels.length).fill(0); const min = new Array(this.levels.length).fill(0);
    const originalRhs = this.levels[0]!.rhs[0];
    const emit = (entryPoint: EntryPoint, sourceIndex: number, destinationIndex = sourceIndex,
      overrides: Partial<GroupResources> = {}, control: readonly [number, number, number, number] = [0, 0, 0, 0],
      dispatchDimensions = this.levels[destinationIndex]!.dimensions) => {
      const source = this.levels[sourceIndex]!, destination = this.levels[destinationIndex]!;
      const defaults: GroupResources = {
        pressureIn: source.pressure[p[sourceIndex]!]!, pressureOut: destination.pressure[p[destinationIndex]! ^ 1]!,
        rhsIn: source.rhs[0], rhsOut: destination.rhs[0], phiIn: source.phi[phi[sourceIndex]!]!,
        phiOut: destination.phi[phi[destinationIndex]! ^ 1]!, volumeIn: source.volume[0], volumeOut: destination.volume[0],
        residualIn: source.residual[0], residualOut: destination.residual[0],
        minimumIn: source.minimum[min[sourceIndex]!]!, minimumOut: destination.minimum[min[destinationIndex]! ^ 1]!,
        coefficientsIn: source.coefficients, coefficientsOut: destination.coefficients,
      };
      const resources = { ...defaults, ...overrides };
      // Indexed by binding, so neither this nor the alias test below allocates
      // per dispatch: the walk runs them thousands of times per re-plan.
      const texturesByBinding: (GPUTexture | undefined)[] = [];
      texturesByBinding[1] = resources.pressureIn; texturesByBinding[2] = resources.pressureOut;
      texturesByBinding[3] = resources.rhsIn; texturesByBinding[4] = resources.rhsOut;
      texturesByBinding[5] = resources.phiIn; texturesByBinding[6] = resources.phiOut;
      texturesByBinding[7] = resources.volumeIn; texturesByBinding[8] = resources.volumeOut;
      texturesByBinding[9] = resources.residualIn; texturesByBinding[10] = resources.residualOut;
      texturesByBinding[11] = resources.minimumIn; texturesByBinding[12] = resources.minimumOut;
      texturesByBinding[14] = resources.coefficientsIn; texturesByBinding[15] = resources.coefficientsOut;
      texturesByBinding[16] = resources.pressureRW;
      for (const sampledBinding of ENTRY_SAMPLED[entryPoint]) {
        const texture = texturesByBinding[sampledBinding];
        for (const writableBinding of ENTRY_WRITABLE[entryPoint]) {
          if (texture !== undefined && texture === texturesByBinding[writableBinding]) {
            throw new Error(`Uniform CM11a ${entryPoint} aliases a sampled and writable texture`);
          }
        }
      }
      const paperM = this.levels.length;
      const paperDestination = paperM - destinationIndex;
      const [params, paramsKey] = this.parameterBuffer(source.dimensions, destination.dimensions,
        destinationIndex,
        [control[0] || paperDestination, control[1] || paperM - UNIFORM_CM11A_PHI_PRESERVATION_LEVELS,
          control[2], control[3]], recovering ? 2 : (planStage === "full-cycle" || planStage === "v-cycle" ? 1 : 0),texturesByBinding);
      // Only the bindings this entry point declares reach the group, so the
      // key is that filtered list -- two entry points that read the same
      // texture through different bindings must not share a group.
      const bindings = ENTRY_BINDINGS[entryPoint];
      let groupKey = `${entryPoint}|${paramsKey}`;
      for (const binding of bindings) {
        const texture = texturesByBinding[binding];
        if (texture !== undefined) groupKey += `|${binding}:${this.textureId(texture)}`;
      }
      let group = this.groupCache.get(groupKey);
      if (!group) {
        const allEntries: GPUBindGroupEntry[] = [
          { binding: 18, resource: { buffer: this.cycleDispatch } },
          { binding: 17, resource: { buffer: this.toleranceBuffer } },
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: this.viewOf(resources.pressureIn) }, { binding: 2, resource: this.viewOf(resources.pressureOut) },
          { binding: 3, resource: this.viewOf(resources.rhsIn) }, { binding: 4, resource: this.viewOf(resources.rhsOut) },
          { binding: 5, resource: this.viewOf(resources.phiIn) }, { binding: 6, resource: this.viewOf(resources.phiOut) },
          { binding: 7, resource: this.viewOf(resources.volumeIn) }, { binding: 8, resource: this.viewOf(resources.volumeOut) },
          { binding: 9, resource: this.viewOf(resources.residualIn) }, { binding: 10, resource: this.viewOf(resources.residualOut) },
          { binding: 11, resource: this.viewOf(resources.minimumIn) }, { binding: 12, resource: this.viewOf(resources.minimumOut) },
          { binding: 13, resource: { buffer: this.diagnostics } },
          { binding: 14, resource: this.viewOf(resources.coefficientsIn) },
          { binding: 15, resource: this.viewOf(resources.coefficientsOut) },
          ...(resources.pressureRW ? [{ binding: 16, resource: this.viewOf(resources.pressureRW) }] : []),
        ];
        group = this.device.createBindGroup({ label: `Uniform CM11a bindings - ${entryPoint}`,
          layout: this.groupLayouts[entryPoint],
          entries: allEntries.filter(({ binding }) => (binding === 13 || bindings.includes(binding))) });
        this.groupCache.set(groupKey, group);
        this.ownedGroups.push(group);
      }
      result.push({ pipeline: this.pipelines![entryPoint], group,
        entryPoint, stage: planStage,
        activeLevel: destinationIndex,
        cycleGate: !["mgPublishCycleDispatch", "mgCheckCycleConvergence", "mgSaveAccepted", "mgRestoreRejected", "mgFinishSafety"].includes(entryPoint)
          ? (recovering ? 2 : (planStage === "full-cycle" || planStage === "v-cycle" ? 1 : 0)) : 0,
        residualCheckpoint: entryPoint === "mgMeasureFineResidual" && control[2] === 1,
        workgroups: this.pagedStorage && !this.logicalPageDispatch && dispatchDimensions.some(n=>n>1)
          ? uniformPressurePageWorkgroups(dispatchDimensions)
          : [Math.ceil(dispatchDimensions[0] / 4), Math.ceil(dispatchDimensions[1] / 4), Math.ceil(dispatchDimensions[2] / 4)],
        ...(entryPoint === "mgSolveCoarsest" ? { coarsestCapture: {
          invocation: control[2],
          dimensions: source.dimensions, pressure: resources.pressureOut, rhs: resources.rhsIn,
          minimum: resources.minimumIn, phi: resources.phiIn, topology: resources.volumeIn,
        } } : {}),
      });
    };
    const flipPressure = (level: number) => { p[level] ^= 1; };
    const flipMinimum = (level: number) => { min[level] ^= 1; };
    emit("mgBuildFinestTopology", 0, 0, { phiOut: this.levels[0]!.phi[0],
      volumeOut: this.levels[0]!.volume[0] });
    emit("mgBuildFinestRhs", 0, 0, { pressureOut: this.levels[0]!.pressure[0],
      rhsOut: originalRhs, minimumOut: this.levels[0]!.minimum[0] });
    // CM11a Algorithm 1 builds the complete raw phi/V pyramid first. Phi
    // continuation is a separate per-level operation and must never feed the
    // next coarsening step.
    for (let level = 0; level + 1 < this.levels.length; level += 1) {
      emit("mgDownsampleTopology", level, level + 1, {
        phiIn: this.levels[level]!.phi[0], phiOut: this.levels[level + 1]!.phi[0],
        volumeOut: this.levels[level + 1]!.volume[0] });
    }
    for (let level = 0; level < this.levels.length; level += 1) {
      emit("mgExtrapolatePhiOneCell", level, level,
        { phiIn: this.levels[level]!.phi[0], phiOut: this.levels[level]!.phi[1] });
      phi[level] = 1;
      emit("mgBakeCoefficients", level, level, {
        phiIn: this.levels[level]!.phi[1], coefficientsOut: this.levels[level]!.coefficients,
      });
    }
    // CM11a summarizes PRBGS as two colour passes plus a projection, but the
    // smoother projects at write time on both its update and pass-through
    // paths, so each sweep exits already-projected and the third pass is gone.
    const sweep = (level: number, rhs: GPUTexture, quietRows = false) => {
      for (let colour = 0; colour < 2; colour += 1) {
        if (this.inPlaceSmoothing) {
          // Two ping-pong flips are the identity, so leaving the parity alone
          // keeps every later dispatch bound exactly as it was.
          const [nx, ny, nz] = this.levels[level]!.dimensions;
          emit("mgSmoothColourInPlace", level, level,
            { rhsIn: rhs, pressureRW: this.levels[level]!.pressure[p[level]!]! }, [0, 0, colour, 0],
            [Math.ceil(nx / 2), ny, nz]);
          // Same bindings, equivalent layout: the group serves either pipeline.
          if (quietRows && this.inPlaceSmoothing) result[result.length - 1]!.quiet = { pipeline: this.pipelines!.mgSmoothRowInPlace,
            workgroups: [Math.ceil(Math.ceil(nx / 2) / ROW_SEGMENT / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)] };
        } else {
          emit("mgSmoothColour", level, level, { rhsIn: rhs }, [0, 0, colour, 0]); flipPressure(level);
        }
      }
    };
    // A visit's sweeps run back to back with nothing between them, so on a
    // level small enough for one workgroup they are one dispatch.
    const smooth = (level: number, rhs: GPUTexture, sweeps: number) => {
      const [nx, ny, nz] = this.levels[level]!.dimensions;
      // Indirect cycle dispatch sizes every launch from the level's tile record.
      if (this.inPlaceSmoothing && FUSE_VISITS && !this.gpuCycleDispatch && sweeps > 0
        && nx * ny * nz <= FUSED_VISIT_MAX_CELLS) {
        emit("mgSmoothVisitInPlace", level, level,
          { rhsIn: rhs, pressureRW: this.levels[level]!.pressure[p[level]!]! }, [0, 0, sweeps, 0], [1, 1, 1]);
      } else for (let i = 0; i < sweeps; i += 1) sweep(level, rhs);
    };
    let coarseInvocation = 0;
    const coarseSolve = (rhs: GPUTexture) => {
      const level = this.levels.length - 1;
      coarseInvocation += 1;
      emit("mgSolveCoarsest", level, level, { rhsIn: rhs },
        [0, 0, coarseInvocation, UNIFORM_CM11A_COARSE_SWEEP_CAP], [1, 1, 1]);
      flipPressure(level);
    };
    const vCycle = (level: number, rhs: GPUTexture): void => {
      if (level === this.levels.length - 1) { coarseSolve(rhs); return; }
      smooth(level, rhs, this.schedule.preSweeps);
      const residualOut = rhs === this.levels[level]!.residual[0]
        ? this.levels[level]!.rhs[1] : this.levels[level]!.residual[0];
      emit("mgResidual", level, level, { rhsIn: rhs, residualOut });
      emit("mgRestrictResidual", level, level + 1, { residualIn: residualOut, rhsOut: this.levels[level + 1]!.rhs[0] });
      emit("mgClearPressure", level + 1); flipPressure(level + 1);
      // Bound-active residuals must not enter an unconstrained coarse solve.
      emit("mgDownsampleSubtract", level, level + 1); flipMinimum(level + 1);
      vCycle(level + 1, this.levels[level + 1]!.rhs[0]);
      emit("mgProlongateAdd", level + 1, level, { residualIn: this.levels[level]!.pressure[p[level]] }); flipPressure(level);
      smooth(level, rhs, this.schedule.postSweeps);
    };
    const fullCycle = () => {
      // Algorithm 3 requires p_tmp to survive every nested V-cycle. Both
      // finest residual[0] and rhs[1] are selected as residual scratch by
      // vCycle(), so the backup must have dedicated storage.
      const backup = this.fullCycleBackup;
      emit("mgCopyPressure", 0, 0, { pressureOut: backup });
      emit("mgShiftMinimum", 0); flipMinimum(0);
      emit("mgResidual", 0, 0, { rhsIn: originalRhs, residualOut: this.levels[0]!.residual[0] });
      const correctionRhs: GPUTexture[] = [this.levels[0]!.residual[0]];
      for (let level = 0; level + 1 < this.levels.length; level += 1) {
        const nextRhs = this.levels[level + 1]!.rhs[1]; correctionRhs.push(nextRhs);
        emit("mgRestrictResidual", level, level + 1, { residualIn: correctionRhs[level]!, rhsOut: nextRhs });
        emit("mgDownsampleMinimum", level, level + 1); flipMinimum(level + 1);
      }
      const coarse = this.levels.length - 1; emit("mgClearPressure", coarse); flipPressure(coarse);
      coarseSolve(correctionRhs[coarse]!);
      for (let level = coarse - 1; level >= 0; level -= 1) {
        emit("mgProlongateAssign", level + 1, level); flipPressure(level);
        vCycle(level, correctionRhs[level]!);
      }
      emit("mgAddPressure", 0, 0, { residualIn: backup }); flipPressure(0);
      min[0] = 0;
    };
    const checkpoint = () => {
      // Canonicalize before deciding to stop. All later cycle writes are gated,
      // so the final projection always sees the last completed cycle in A.
      if (p[0] !== 0) {
        emit("mgCopyPressure", 0, 0, { pressureOut: this.levels[0]!.pressure[0] }); p[0] = 0;
      }
      emit("mgMeasureFineResidual", 0, 0, { rhsIn: originalRhs }, [0, 0, 1, 0]);
      emit("mgCheckCycleConvergence", 0, 0, {}, [0, 0, recovering ? 4 : planStage === "full-cycle" ? 2 : 3, 0], [1, 1, 1]);
      // These commits run even when the decision has just stopped the solver.
      // Dedicated storage cannot alias Full-Cycle or V-cycle scratch.
      const [nx, ny, nz] = this.levels[0]!.dimensions;
      const quietCommit = (pipeline: GPUComputePipeline) => {
        if (recovering && this.inPlaceSmoothing && ROW_SWEEP !== "off") result[result.length - 1]!.quiet = { pipeline,
          workgroups: [Math.ceil(nx / ROW_SEGMENT / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)] };
      };
      emit("mgSaveAccepted", 0, 0, { pressureOut: this.acceptedPressure });
      quietCommit(this.pipelines!.mgSaveAcceptedQuiet);
      emit("mgRestoreRejected", 0, 0, { pressureOut: this.levels[0]!.pressure[0], residualIn: this.acceptedPressure });
      quietCommit(this.pipelines!.mgRestoreRejectedQuiet);
      if (this.gpuCycleDispatch) emit("mgPublishCycleDispatch", 0, 0, {}, [0, 0, this.levels.length + 1, 0], [1, 1, 1]);
    };
    // Where a lagged budget may cut. Entry 0 is the end of setup; entry k is
    // the end of cycle k, which is always a checkpoint.
    emit("mgMeasureFineResidual", 0, 0, { rhsIn: originalRhs }, [0, 0, 1, 0]);
    emit("mgCheckCycleConvergence", 0, 0, {}, [0, 0, 0, 0], [1, 1, 1]);
    emit("mgCopyPressure", 0, 0, { pressureOut: this.acceptedPressure });
    if (this.gpuCycleDispatch) emit("mgPublishCycleDispatch", 0, 0, {}, [0, 0, this.levels.length + 1, 0], [1, 1, 1]);
    const cycleBoundaries: number[] = [result.length];
    planStage = "full-cycle";
    for (let cycle = 0; cycle < this.schedule.fullCycles; cycle += 1) {
      fullCycle(); checkpoint(); cycleBoundaries.push(result.length);
      yield;
    }
    planStage = "v-cycle";
    for (let cycle = 0; cycle < this.schedule.vCycles; cycle += 1) {
      vCycle(0, originalRhs); checkpoint(); cycleBoundaries.push(result.length);
      yield;
    }
    planStage = "finish";
    this.cycleBoundaries = Object.freeze(cycleBoundaries);
    this.finishStart = result.length;
    recovering = true;
    for (let batch = 0; batch < UNIFORM_CM11A_RECOVERY_BATCHES; batch += 1) {
      for (let i = 0; i < UNIFORM_CM11A_RECOVERY_SWEEPS; i += 1) sweep(0, originalRhs, ROW_SWEEP !== "off");
      checkpoint();
    }
    recovering = false;
    emit("mgRestoreRejected", 0, 0, { pressureOut: this.levels[0]!.pressure[0], residualIn: this.acceptedPressure }, [0, 0, 0, 1]);
    emit("mgFinishSafety", 0, 0, {}, [0, 0, 0, 0], [1, 1, 1]);
    if (p[0] !== 0) { emit("mgCopyPressure", 0, 0, { pressureOut: this.levels[0]!.pressure[0] }); p[0] = 0; }
    emit("mgMeasureFineResidual", 0, 0, {
      pressureIn: this.levels[0]!.pressure[0], rhsIn: originalRhs,
      phiIn: this.levels[0]!.phi[phi[0]], volumeIn: this.levels[0]!.volume[0],
      minimumIn: this.levels[0]!.minimum[min[0]],
    });
    return result;
  }

  private buildPlan(): PlannedDispatch[] {
    const steps = this.buildPlanSteps();
    for (;;) { const next = steps.next(); if (next.done) return next.value; }
  }

  /**
   * Spend up to `budgetMs` on a deferred plan, and report whether it is ready.
   *
   * A budget of zero still advances one cycle, so a caller cannot livelock the
   * build by asking for nothing. Ready instances answer true without work.
   */
  advancePlan(budgetMs: number): boolean {
    if (this.plan) return true;
    if (!this.pipelines) return false;
    this.planSteps ??= this.buildPlanSteps();
    const started = typeof performance !== "undefined" ? performance.now() : 0;
    for (;;) {
      const next = this.planSteps.next();
      if (next.done) {
        this.plan = Object.freeze(next.value); this.planSteps = undefined; return true;
      }
      if ((typeof performance !== "undefined" ? performance.now() : 0) - started >= budgetMs) return false;
    }
  }

  /** Whether this instance is still being built a cycle at a time. */
  get planPending(): boolean { return this.plan === undefined && this.planSteps !== undefined; }

  /**
   * The 80 uniform bytes a dispatch reads, and a key for them. Dimensions,
   * level and control repeat across every cycle, so a plan of thousands needs
   * only a few dozen distinct buffers.
   */
  private parameterBuffer(level: readonly [number, number, number], coarse: readonly [number, number, number],
    activeLevel: number, control: readonly [number, number, number, number],
    gated: number, texturesByBinding: readonly (GPUTexture|undefined)[]): [GPUBuffer, string] {
    const bytes = new ArrayBuffer(this.scratchFields?352:this.pagedStorage?336:80); const u = new Uint32Array(bytes); const f = new Float32Array(bytes);
    u.set(this.levels[0]!.dimensions, 0); u.set(level, 4); u.set(coarse, 8);
    // Each axis has coarsened by however many times *it* was halved, which is
    // no longer one shared 2**levelIndex once a hierarchy is semi-coarsened.
    // Reading the factor back off the lattice keeps the two in step by
    // construction, and reproduces 2**levelIndex exactly when they are equal.
    f.set(this.spacing.map((value, axis) =>
      value * (this.finestSize[axis]! / (level[axis]! - 2))), 12);
    u.set(control, 16);
    u[3] = activeLevel;
    u[7] = gated;
    if(this.pagedStorage||this.scratchFields)texturesByBinding.forEach((texture,binding)=>{
      if(texture){u.set(this.logicalDimensions.get(texture)!,20+4*binding);u[23+4*binding]=this.scratchFields?.scratchMetadata(texture)??0;}
    });
    const dimensionsKey=this.pagedStorage||this.scratchFields?Array.from(u.subarray(20)).join(","):"";
    const key = `${dimensionsKey}|${level.join(",")}|${coarse.join(",")}|${activeLevel}|${control.join(",")}|${gated}`;
    const cached = this.paramCache.get(key);
    if (cached) return [cached, key];
    const buffer = this.device.createBuffer({ label: "Uniform CM11a dispatch parameters", size: bytes.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, bytes); this.ownedParams.push(buffer);
    this.paramCache.set(key, buffer);
    return [buffer, key];
  }

  destroy(): void { if (this.isDestroyed) return; this.isDestroyed = true;
    for (const level of this.levels) for (const pair of [level.pressure, level.rhs, level.phi,
      level.volume, level.residual, level.minimum]) {
      for (const texture of new Set(pair)) texture.destroy();
    }
    for (const level of this.levels) level.coefficients.destroy();
    for(const buffer of this.pageCaptureBuffers)buffer.destroy();
    this.pressurePublication?.destroy();
    this.cycleDispatch.destroy();
    this.fullCycleBackup.destroy();
    this.acceptedPressure.destroy();
    this.toleranceBuffer.destroy();
    for (const buffer of this.ownedParams) buffer.destroy(); this.diagnostics.destroy();
    if (this.coarsestCaptureBuffers) for (const buffer of [this.coarsestCaptureBuffers.pressure,
      this.coarsestCaptureBuffers.rhs, this.coarsestCaptureBuffers.minimum,
      this.coarsestCaptureBuffers.phi, this.coarsestCaptureBuffers.topology]) buffer.destroy();
    this.ownedParams.length = 0; this.ownedGroups.length = 0; this.plan = undefined; this.pipelines = undefined;
    this.viewCache.clear(); this.paramCache.clear(); this.groupCache.clear(); this.textureIds.clear();
    this.planSteps = undefined;
  }
  private assertLive(): void { if (this.isDestroyed) throw new Error("Uniform CM11a hierarchy is destroyed"); }
}
