import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE,
  uniformPressureMultigridWGSL } from "./webgpu-uniform-pressure-multigrid.wgsl";

export const UNIFORM_CM11A_FULL_CYCLES = 3;
export const UNIFORM_CM11A_V_CYCLES = 4;
// CM11a measured four pre/post sweeps on its published grids. Six is the
// minimum robust schedule for the deeper 64x32x64 hierarchy: at four or five,
// a late Full-Cycle coarse correction reaches the 4096-sweep cap and its
// unconverged pressure injects 10^7 m/s into the first projection.
export const UNIFORM_CM11A_PRE_SWEEPS = 6;
export const UNIFORM_CM11A_POST_SWEEPS = 6;
export const UNIFORM_CM11A_CONSTRAINT_LEVELS = 3;
export const UNIFORM_CM11A_PHI_PRESERVATION_LEVELS = 2;
// TallCells reports 1e-4 s^-1 as its GPU/single-precision absolute L-infinity
// tolerance; 1e-8 belongs to its double-precision CPU comparison. CM11a
// itself fixes the cycle schedule but does not prescribe a residual tolerance.
export { UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE };
export const UNIFORM_CM11A_COARSE_SWEEP_CAP = 4096;

export interface UniformCM11aSchedule {
  readonly fullCycles: number;
  readonly vCycles: number;
  readonly preSweeps: number;
  readonly postSweeps: number;
  readonly residualTolerance?: number;
}

export const DEFAULT_UNIFORM_CM11A_SCHEDULE: UniformCM11aSchedule = Object.freeze({
  fullCycles: UNIFORM_CM11A_FULL_CYCLES,
  vCycles: UNIFORM_CM11A_V_CYCLES,
  preSweeps: UNIFORM_CM11A_PRE_SWEEPS,
  postSweeps: UNIFORM_CM11A_POST_SWEEPS,
  residualTolerance: 10,
});

/**
 * Cycles the lagged budget never drops below. One complete cycle always runs,
 * so a step whose demand estimate is stale by a frame still projects against a
 * coarse-corrected pressure rather than against the previous step's field.
 */
export const UNIFORM_CM11A_MINIMUM_CYCLE_BUDGET = 1;
/** Cycles added above the last observed demand when it converged. */
export const UNIFORM_CM11A_DEFAULT_BUDGET_HEADROOM = 1;

export interface UniformCM11aCycleBudgetInput {
  /**
   * Cycles the latest *observed* step executed before its residual gate
   * tripped. Undefined until the first asynchronous diagnostics sample lands.
   */
  readonly lastExecutedCycles?: number;
  /** Whether that step met the tolerance; false means it ran to its ceiling. */
  readonly lastConverged?: boolean;
  readonly headroom: number;
  readonly minCycles?: number;
  /** The configured schedule: Full-Cycles + V-Cycles actually planned. */
  readonly maxCycles: number;
}

/**
 * How many cycles the next step encodes, from the last demand the async stats
 * readback reported.
 *
 * The GPU-side gate already stops a converged solve early, but a skipped pass
 * still costs its launch floor and its CPU encode, so the saving has to be
 * taken on the host by not encoding the tail at all. The signal is lagged by
 * however many frames the readback takes, which is why the rule is asymmetric:
 * shrinking is capped at one cycle of headroom above observed demand, while a
 * step that used every encoded cycle *and still missed tolerance* doubles, so
 * an impact frame recovers its full schedule within one or two steps instead
 * of climbing one cycle at a time.
 */
export function uniformCM11aCycleBudget(input: UniformCM11aCycleBudgetInput): number {
  const maxCycles = Number.isFinite(input.maxCycles) ? Math.max(0, Math.floor(input.maxCycles)) : 0;
  const minCycles = Math.min(maxCycles, Math.max(0, Math.floor(
    Number.isFinite(input.minCycles) ? input.minCycles! : UNIFORM_CM11A_MINIMUM_CYCLE_BUDGET)));
  const executed = input.lastExecutedCycles;
  // No sample yet: encode the configured schedule, which is exactly what the
  // solver did before this rule existed.
  if (executed === undefined || !Number.isFinite(executed)) return maxCycles;
  const observed = Math.max(0, Math.floor(executed));
  const headroom = Number.isFinite(input.headroom) ? Math.max(0, Math.floor(input.headroom)) : 0;
  const demand = input.lastConverged
    ? observed + headroom
    : Math.max(2 * observed, observed + 2);
  return Math.min(maxCycles, Math.max(minCycles, demand));
}

const ENTRY_POINTS = [
  "mgBuildFinestTopology", "mgBuildFinestRhs", "mgDownsampleTopology", "mgExtrapolatePhiOneCell",
  "mgBakeCoefficients",
  "mgResidual", "mgRestrictResidual", "mgProlongateAdd", "mgProlongateAssign",
  "mgDownsampleSubtract", "mgDownsampleMinimum", "mgSmoothColour",
  "mgCopyPressure", "mgClearPressure", "mgClearMinimum",
  "mgShiftMinimum", "mgAddPressure", "mgSolveCoarsest", "mgMeasureFineResidual", "mgCheckCycleConvergence",
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
  mgCopyPressure: [0, 1, 2], mgClearPressure: [0, 2], mgClearMinimum: [0, 12],
  mgShiftMinimum: [0, 1, 11, 12], mgAddPressure: [0, 1, 2, 9],
  mgSolveCoarsest: [0, 1, 2, 3, 5, 7, 11, 13],
  mgMeasureFineResidual: [0, 1, 3, 11, 13, 14, 17],
  mgCheckCycleConvergence: [0, 17],
});

/**
 * The same split, per entry point and computed once. The plan walk runs this
 * test thousands of times and rebuilding the sets and filtered lists inside it
 * was a measurable share of a re-plan.
 */
const SAMPLED_BINDINGS: readonly number[] = [1, 3, 5, 7, 9, 11, 14];
const WRITABLE_BINDINGS: readonly number[] = [2, 4, 6, 8, 10, 12, 15];
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

/** The 256-lane workgroup that solves the coarsest level exactly. */
export const UNIFORM_CM11A_COARSEST_LANES = 256;

/** Physical cells on each axis, halo excluded. */
export type UniformCM11aLevelSize = readonly [number, number, number];

/** What a dense CM11a hierarchy over `dimensions` would look like. */
export interface UniformCM11aHierarchyPlan {
  readonly levelCount: number;
  /** Physical lattice of every level, finest first. */
  readonly levelDimensions: readonly UniformCM11aLevelSize[];
  /** Coarsest lattice with its one-cell halo, which the 256 lanes cover. */
  readonly coarsestCells: number;
  /** True when each axis coarsens on its own schedule. */
  readonly semiCoarsened: boolean;
  /** Why the hierarchy is impossible, or `undefined` when it is buildable. */
  readonly rejection?: string;
}

/** Alignments a window capacity and origin may use, widest first. */
export const UNIFORM_CM11A_WINDOW_ALIGNMENTS = [32, 16] as const;

export interface UniformCM11aWindowPlan {
  readonly capacity: UniformCM11aLevelSize;
  readonly origin: readonly [number, number, number];
  readonly hierarchy: UniformCM11aHierarchyPlan;
  /** Alignment used per axis; a domain-wide axis reports its own length. */
  readonly alignment: readonly [number, number, number];
}

/**
 * Where the current capacity can sit so that it covers `[low, high)`.
 *
 * Returns the aligned origin, or undefined when the capacity is simply too
 * small. The caller uses this to keep an instance across a step in which the
 * liquid moved but did not grow -- which is most steps.
 */
export function seatUniformCM11aWindow(
  domain: UniformCM11aLevelSize,
  capacity: readonly [number, number, number],
  alignment: readonly [number, number, number],
  low: readonly [number, number, number],
  high: readonly [number, number, number],
): [number, number, number] | undefined {
  const origin: [number, number, number] = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    const step = alignment[axis]!, size = domain[axis]!, width = capacity[axis]!;
    if (width > size) return undefined;
    const seat = Math.max(0, Math.min(Math.floor(Math.max(0, low[axis]!) / step) * step, size - width));
    if (seat > low[axis]! || seat + width < high[axis]!) return undefined;
    origin[axis] = seat;
  }
  return origin;
}

/**
 * Capacity and origin of a CM11a lattice covering `[low, high)`.
 *
 * Aligning both keeps every coarse grid registered to the domain, so the
 * origin only has to move when the liquid crosses an alignment boundary, and
 * the same capacity is reached again and again instead of drifting by a cell
 * a step. The lockstep hierarchy is preferred over semi-coarsening for the
 * same reason the domain planner prefers it, so a capacity one alignment step
 * wider, or a shortest axis lifted to a power of two, is tried before giving
 * up on it.
 */
export function planUniformCM11aWindow(
  domain: UniformCM11aLevelSize,
  low: readonly [number, number, number],
  high: readonly [number, number, number],
): UniformCM11aWindowPlan {
  const alignment = domain.map((size) =>
    UNIFORM_CM11A_WINDOW_ALIGNMENTS.find((step) => size % step === 0 && size > step) ?? size,
  ) as unknown as [number, number, number];
  const seat = (request: readonly number[]): { capacity: UniformCM11aLevelSize;
    origin: [number, number, number] } => {
    const capacity: number[] = []; const origin: number[] = [];
    for (let axis = 0; axis < 3; axis += 1) {
      const step = alignment[axis]!, size = domain[axis]!;
      const width = Math.min(size, Math.max(step, Math.ceil(request[axis]! / step) * step));
      capacity.push(width);
      origin.push(Math.max(0, Math.min(
        Math.floor(Math.max(0, low[axis]!) / step) * step, size - width)));
    }
    return { capacity: capacity as unknown as UniformCM11aLevelSize,
      origin: origin as [number, number, number] };
  };
  const requested = [0, 1, 2].map((axis) => {
    const step = alignment[axis]!;
    const start = Math.floor(Math.max(0, low[axis]!) / step) * step;
    return Math.max(2, Math.min(domain[axis]!, Math.max(high[axis]! - start, 0)));
  });
  const candidates: number[][] = [requested];
  for (let axis = 0; axis < 3; axis += 1) {
    const bumped = [...requested];
    bumped[axis] = Math.min(domain[axis]!, bumped[axis]! + alignment[axis]!);
    candidates.push(bumped);
  }
  const lifted = [...requested];
  const shortest = lifted.indexOf(Math.min(...lifted));
  lifted[shortest] = Math.min(domain[shortest]!,
    2 ** Math.ceil(Math.log2(Math.max(2, lifted[shortest]!))));
  candidates.push(lifted, [...domain]);
  let fallback: UniformCM11aWindowPlan | undefined;
  for (const candidate of candidates) {
    const { capacity, origin } = seat(candidate);
    const hierarchy = planUniformCM11aHierarchy(capacity);
    if (hierarchy.rejection) continue;
    const plan: UniformCM11aWindowPlan = { capacity, origin, hierarchy, alignment };
    if (!hierarchy.semiCoarsened) return plan;
    fallback ??= plan;
  }
  return fallback ?? { capacity: domain, origin: [0, 0, 0],
    hierarchy: planUniformCM11aHierarchy(domain), alignment };
}

const haloedCells = (size: UniformCM11aLevelSize): number =>
  size.reduce((cells, value) => cells * (value + 2), 1);

/**
 * The original hierarchy: halve all three axes together, stopping when the
 * thinnest reaches two cells. Returns undefined when it cannot be built.
 */
function lockstepLevels(dimensions: UniformCM11aLevelSize): UniformCM11aLevelSize[] | undefined {
  const minimum = Math.min(...dimensions);
  if ((minimum & (minimum - 1)) !== 0) return undefined;
  const levelCount = Math.floor(Math.log2(minimum));
  const coarsening = 2 ** (levelCount - 1);
  if (!dimensions.every((value) => value % coarsening === 0)) return undefined;
  const levels: UniformCM11aLevelSize[] = [];
  for (let index = 0; index < levelCount; index += 1) {
    const step = 2 ** index;
    levels.push([dimensions[0] / step, dimensions[1] / step, dimensions[2] / step]);
  }
  return haloedCells(levels[levels.length - 1]!) <= UNIFORM_CM11A_COARSEST_LANES ? levels : undefined;
}

/**
 * Semi-coarsening: halve each axis on its own schedule, while it is even and
 * still above two cells. An axis that has bottomed out simply stops, and the
 * others keep going.
 */
function semiCoarsenedLevels(dimensions: UniformCM11aLevelSize): UniformCM11aLevelSize[] {
  const levels: UniformCM11aLevelSize[] = [dimensions];
  for (;;) {
    const previous = levels[levels.length - 1]!;
    const next = previous.map((value) =>
      value % 2 === 0 && value > 2 ? value / 2 : value) as unknown as UniformCM11aLevelSize;
    if (next.every((value, axis) => value === previous[axis])) return levels;
    levels.push(next);
  }
}

/**
 * Whether a lattice can carry the dense hierarchy, without a GPU.
 *
 * This is a *constructor* precondition, not a scene-validity one: a scene can
 * satisfy every rule in `validateScene`, build, render its dry world and still
 * fail to load the instant the solver is created. Exposing it as arithmetic is
 * what lets the catalog be checked on the CPU.
 *
 * Two rules, tried in order, and the order is the whole point. The original
 * hierarchy coarsens all three axes in **lockstep** and stops when the thinnest
 * reaches two cells, so a thin axis caps how far the wide ones may coarsen: a
 * 128x128x8 lattice reaches only 32x32x2 and leaves 4624 coarsest cells for 256
 * lanes. Coarsening each axis on its own schedule reaches 2x2x2 instead --
 * fewer coarsest cells, not more, because the 256 lanes were never the real
 * limit. But semi-coarsening also builds a *different* hierarchy wherever both
 * rules apply, and the D4 folds downstream make that a change in rounding, so
 * the lockstep plan is preferred whenever it exists and semi-coarsening is
 * reached only by lattices that have no hierarchy at all today. Every scene
 * that loads now keeps the hierarchy, and the numbers, that it already has.
 */
export function planUniformCM11aHierarchy(
  dimensions: UniformCM11aLevelSize,
): UniformCM11aHierarchyPlan {
  const reject = (rejection: string): UniformCM11aHierarchyPlan =>
    ({ levelCount: 0, levelDimensions: [], coarsestCells: 0, semiCoarsened: false, rejection });
  if (!dimensions.every((value) => Number.isSafeInteger(value) && value >= 2)) {
    return reject("CM11a dense hierarchy requires positive integral dimensions of at least two cells");
  }
  const lockstep = lockstepLevels(dimensions);
  const levels = lockstep ?? semiCoarsenedLevels(dimensions);
  const coarsestCells = haloedCells(levels[levels.length - 1]!);
  const plan = {
    levelCount: levels.length, levelDimensions: Object.freeze(levels),
    coarsestCells, semiCoarsened: lockstep === undefined,
  };
  if (coarsestCells > UNIFORM_CM11A_COARSEST_LANES) {
    return { ...plan,
      rejection: "CM11a coarsest grid must be integral and fit its 256-lane high-precision solve" };
  }
  return plan;
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
}

/**
 * Dense uniform implementation of CM11a Algorithms 1--3. The shader source
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
  readonly module: GPUShaderModule;
  readonly groupLayouts: Readonly<Record<string, GPUBindGroupLayout>>;
  readonly pipelines: Readonly<Record<string, GPUComputePipeline>>;
}

export class WebGPUUniformPressureMultigrid {
  readonly levels: readonly UniformPressureMultigridLevel[];
  readonly shaderFragment = uniformPressureMultigridWGSL;
  readonly diagnostics: GPUBuffer;
  private readonly toleranceBuffer: GPUBuffer;
  readonly allocatedBytes: number;
  /** CM11a Algorithm 3 p_tmp; no V-cycle scratch dispatch may alias it. */
  private readonly fullCycleBackup: GPUTexture;
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
    deferPlan = false) {
    const hierarchy = planUniformCM11aHierarchy(
      dimensions as readonly [number, number, number]);
    if (hierarchy.rejection) throw new RangeError(hierarchy.rejection);
    if (!spacing.every((value) => Number.isFinite(value) && value > 0)) {
      throw new RangeError("CM11a grid spacing must be positive and finite");
    }
    this.spacing = spacing;
    const { levelCount } = hierarchy;
    this.finestSize = hierarchy.levelDimensions[0]!;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    let allocatedBytes = 92;
    const texture = (label: string, format: GPUTextureFormat,
      size: readonly [number, number, number]) => {
      const result = device.createTexture({ label, size: [...size], dimension: "3d", format, usage });
      allocatedBytes += size[0] * size[1] * size[2] * (format === "rgba32float" ? 16 : 4);
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
      levels.push(Object.freeze({ dimensions: size, pressure: pair("pressure"), rhs: pair("rhs"),
        phi: pair("phi"), volume: pair("V", "rgba32float"), residual: pair("residual"),
        minimum: pair("p-min"), coefficients: texture(`Uniform CM11a L${index} coefficients`, "rgba32float", size) }));
    }
    this.levels = Object.freeze(levels);
    this.fullCycleBackup = texture("Uniform CM11a Full-Cycle p_tmp", "r32float", levels[0]!.dimensions);
    this.diagnostics = device.createBuffer({ label: "Uniform CM11a convergence status", size: 76,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.toleranceBuffer = device.createBuffer({ label: "Pressure residual tolerance", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.setResidualTolerance(schedule.residualTolerance ?? 10);
    this.allocatedBytes = allocatedBytes;
    const textureBinding = { sampleType: "unfilterable-float", viewDimension: "3d" } as const;
    const scalarStorage = { access: "write-only", format: "r32float", viewDimension: "3d" } as const;
    const allEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ...[1, 3, 5, 7, 9, 11].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, texture: textureBinding })),
      ...[2, 4, 6, 10, 12].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, storageTexture: scalarStorage })),
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 14, visibility: GPUShaderStage.COMPUTE, texture: textureBinding },
      { binding: 15, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
    ];
    this.groupLayouts = (programs?.groupLayouts as Record<EntryPoint, GPUBindGroupLayout> | undefined)
      ?? Object.freeze(Object.fromEntries(ENTRY_POINTS.map((entryPoint) => [entryPoint,
        device.createBindGroupLayout({ label: `Uniform CM11a hierarchy layout - ${entryPoint}`,
          entries: allEntries.filter(({ binding }) => (binding === 13 || ENTRY_BINDINGS[entryPoint].includes(binding))) }),
      ])) as Record<EntryPoint, GPUBindGroupLayout>);
    // Given another instance's compiled programs this one is ready here, with
    // no await anywhere: a re-plan happens inside a single step.
    if (programs) {
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
    if (!this.shaderModule || !this.pipelines) return undefined;
    return { module: this.shaderModule, groupLayouts: this.groupLayouts, pipelines: this.pipelines };
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

  get pressureTexture(): GPUTexture { return this.levels[0]!.pressure[0]; }

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
    const entries = await Promise.all(ENTRY_POINTS.map(async (entryPoint) => [entryPoint,
      await compiler.compileComputePipeline({ label: `Uniform CM11a - ${entryPoint}`,
        layout: this.device.createPipelineLayout({ label: `Uniform CM11a layout - ${entryPoint}`,
          bindGroupLayouts: [input.uniformBindGroupLayout, this.groupLayouts[entryPoint]] }),
        compute: { module: shaderModule, entryPoint } }, { priority: "visible", signal: input.signal })] as const));
    this.pipelines = Object.freeze(Object.fromEntries(entries) as Record<EntryPoint, GPUComputePipeline>);
    this.plan = Object.freeze(this.buildPlan());
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
  ): void {
    this.assertLive(); if (!this.plan) throw new Error("Uniform CM11a hierarchy is not initialized");
    encoder.clearBuffer(this.diagnostics);
    const prefixEnd = this.cycleBoundaries?.[this.clampCycleBudget(cycleBudget)] ?? this.plan.length;
    let openStage: UniformCM11aPlanStage | undefined;
    for (let index = 0; index < this.plan.length; index += 1) {
      const dispatch = this.plan[index]!;
      if (openStage !== undefined && dispatch.stage !== openStage) boundary?.(openStage);
      openStage = dispatch.stage;
      // A truncated cycle encodes nothing at all. Its stage seam is still
      // reported above, in order, so the advance's phase partition keeps all
      // four sections and a section with no passes reads as zero-length
      // instead of vanishing from the trace.
      if (index >= prefixEnd && index < this.finishStart) continue;
      // A WebGPU texture usage scope spans the whole compute pass. End the
      // pass between hierarchy stages so storage outputs can become sampled
      // inputs in the next stage.
      if (dispatch.entryPoint === "mgMeasureFineResidual" && dispatch.stage !== "finish") {
        encoder.clearBuffer(this.diagnostics, 60, 4);
      }
      const pass = encoder.beginComputePass({ label: `Uniform CM11a ${dispatch.entryPoint}` });
      pass.setPipeline(dispatch.pipeline); pass.setBindGroup(1, dispatch.group);
      pass.setBindGroup(0, uniformGroup);
      if (this.windowLattice) {
        pass.dispatchWorkgroups(...dispatch.workgroups);
      } else if (this.activeDispatch && dispatch.entryPoint !== "mgSolveCoarsest" && dispatch.entryPoint !== "mgCheckCycleConvergence") {
        const chosen = this.windowLevelGroups?.[dispatch.activeLevel];
        if (chosen) {
          pass.dispatchWorkgroups(Math.min(chosen[0], dispatch.workgroups[0]),
            Math.min(chosen[1], dispatch.workgroups[1]), Math.min(chosen[2], dispatch.workgroups[2]));
        } else {
          const indirectOffset = (16 + dispatch.activeLevel * 10 + 3) * 4;
          pass.dispatchWorkgroupsIndirect(this.activeDispatch, indirectOffset);
        }
      } else {
        pass.dispatchWorkgroups(...dispatch.workgroups);
      }
      pass.end();
      if (dispatch.coarsestCapture && this.coarsestCaptureBuffers
        && dispatch.coarsestCapture.invocation === this.coarsestCaptureBuffers.invocation) {
        const capture = dispatch.coarsestCapture, buffers = this.coarsestCaptureBuffers;
        const destination = (buffer: GPUBuffer) => ({ buffer, bytesPerRow: 256,
          rowsPerImage: capture.dimensions[1] });
        encoder.copyTextureToBuffer({ texture: capture.pressure }, destination(buffers.pressure), capture.dimensions);
        encoder.copyTextureToBuffer({ texture: capture.rhs }, destination(buffers.rhs), capture.dimensions);
        encoder.copyTextureToBuffer({ texture: capture.minimum }, destination(buffers.minimum), capture.dimensions);
        encoder.copyTextureToBuffer({ texture: capture.phi }, destination(buffers.phi), capture.dimensions);
        encoder.copyTextureToBuffer({ texture: capture.topology }, destination(buffers.topology), capture.dimensions);
      }
    }
    if (openStage !== undefined) boundary?.(openStage);
  }

  get levelCount(): number { return this.levels.length; }

  /** Cycles the built plan carries: Full-Cycles plus V-Cycles. */
  get cycleCount(): number { return Math.max(0, (this.cycleBoundaries?.length ?? 1) - 1); }

  /** Compute passes the whole plan encodes. Undefined until `initialize`. */
  get planPassCount(): number | undefined { return this.plan?.length; }

  /** Compute passes a given cycle budget encodes, setup and finish included. */
  encodedPassCount(cycleBudget?: number): number | undefined {
    if (!this.plan || !this.cycleBoundaries) return undefined;
    return this.cycleBoundaries[this.clampCycleBudget(cycleBudget)]!
      + (this.plan.length - this.finishStart);
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
    return counts;
  }

  /** Opt-in diagnostic capture of one coarsest solve invocation. It does not alter the solve. */
  enableCoarsestCapture(invocation = 1): void {
    this.assertLive(); if (this.coarsestCaptureBuffers) return;
    if (!Number.isSafeInteger(invocation) || invocation < 1) {
      throw new RangeError("CM11a coarsest capture invocation must be a positive integer");
    }
    const dimensions = this.levels.at(-1)!.dimensions;
    const byteLength = 256 * dimensions[1] * dimensions[2];
    const buffer = (label: string) => this.device.createBuffer({ label, size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.coarsestCaptureBuffers = { invocation, dimensions, byteLength,
      pressure: buffer("Uniform CM11a capture pressure"), rhs: buffer("Uniform CM11a capture rhs"),
      minimum: buffer("Uniform CM11a capture p-min"), phi: buffer("Uniform CM11a capture phi"),
      topology: buffer("Uniform CM11a capture topology") };
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
          const row = new Float32Array(bytes.buffer, bytes.byteOffset + 256 * (y + dy * z), 64);
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
          control[2], control[3]], planStage === "full-cycle" || planStage === "v-cycle");
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
        workgroups: [Math.ceil(dispatchDimensions[0] / 4), Math.ceil(dispatchDimensions[1] / 4),
          Math.ceil(dispatchDimensions[2] / 4)],
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
    const sweep = (level: number, rhs: GPUTexture) => {
      for (let colour = 0; colour < 2; colour += 1) {
        emit("mgSmoothColour", level, level, { rhsIn: rhs }, [0, 0, colour, 0]); flipPressure(level);
      }
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
      for (let i = 0; i < this.schedule.preSweeps; i += 1) sweep(level, rhs);
      const residualOut = rhs === this.levels[level]!.residual[0]
        ? this.levels[level]!.rhs[1] : this.levels[level]!.residual[0];
      emit("mgResidual", level, level, { rhsIn: rhs, residualOut });
      emit("mgRestrictResidual", level, level + 1, { residualIn: residualOut, rhsOut: this.levels[level + 1]!.rhs[0] });
      emit("mgClearPressure", level + 1); flipPressure(level + 1);
      if (level < UNIFORM_CM11A_CONSTRAINT_LEVELS) {
        emit("mgDownsampleSubtract", level, level + 1); flipMinimum(level + 1);
      } else { emit("mgClearMinimum", level + 1); flipMinimum(level + 1); }
      vCycle(level + 1, this.levels[level + 1]!.rhs[0]);
      emit("mgProlongateAdd", level + 1, level, { residualIn: this.levels[level]!.pressure[p[level]] }); flipPressure(level);
      for (let i = 0; i < this.schedule.postSweeps; i += 1) sweep(level, rhs);
    };
    const fullCycle = () => {
      // Algorithm 3 requires p_tmp to survive every nested V-cycle. Both
      // finest residual[1] and rhs[1] are selected as residual scratch by
      // vCycle(), so the backup must have dedicated storage.
      const backup = this.fullCycleBackup;
      emit("mgCopyPressure", 0, 0, { pressureOut: backup });
      emit("mgShiftMinimum", 0); flipMinimum(0);
      emit("mgResidual", 0, 0, { rhsIn: originalRhs, residualOut: this.levels[0]!.residual[0] });
      const correctionRhs: GPUTexture[] = [this.levels[0]!.residual[0]];
      for (let level = 0; level + 1 < this.levels.length; level += 1) {
        const nextRhs = this.levels[level + 1]!.rhs[1]; correctionRhs.push(nextRhs);
        emit("mgRestrictResidual", level, level + 1, { residualIn: correctionRhs[level]!, rhsOut: nextRhs });
        if (level < UNIFORM_CM11A_CONSTRAINT_LEVELS) {
          emit("mgDownsampleMinimum", level, level + 1); flipMinimum(level + 1);
        } else { emit("mgClearMinimum", level + 1); flipMinimum(level + 1); }
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
      emit("mgCheckCycleConvergence", 0, 0, {}, [0, 0, planStage === "full-cycle" ? 2 : 3, 0], [1, 1, 1]);
    };
    // Where a lagged budget may cut. Entry 0 is the end of setup; entry k is
    // the end of cycle k, which is always a checkpoint.
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
    gated: boolean): [GPUBuffer, string] {
    const bytes = new ArrayBuffer(80); const u = new Uint32Array(bytes); const f = new Float32Array(bytes);
    u.set(this.levels[0]!.dimensions, 0); u.set(level, 4); u.set(coarse, 8);
    // Each axis has coarsened by however many times *it* was halved, which is
    // no longer one shared 2**levelIndex once a hierarchy is semi-coarsened.
    // Reading the factor back off the lattice keeps the two in step by
    // construction, and reproduces 2**levelIndex exactly when they are equal.
    f.set(this.spacing.map((value, axis) =>
      value * (this.finestSize[axis]! / (level[axis]! - 2))), 12);
    u.set(control, 16);
    u[3] = activeLevel;
    u[7] = gated ? 1 : 0;
    const key = `${level.join(",")}|${coarse.join(",")}|${activeLevel}|${control.join(",")}|${gated ? 1 : 0}`;
    const cached = this.paramCache.get(key);
    if (cached) return [cached, key];
    const buffer = this.device.createBuffer({ label: "Uniform CM11a dispatch parameters", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, bytes); this.ownedParams.push(buffer);
    this.paramCache.set(key, buffer);
    return [buffer, key];
  }

  destroy(): void { if (this.isDestroyed) return; this.isDestroyed = true;
    for (const level of this.levels) for (const pair of [level.pressure, level.rhs, level.phi,
      level.volume, level.residual, level.minimum]) { pair[0].destroy(); pair[1].destroy(); }
    for (const level of this.levels) level.coefficients.destroy();
    this.fullCycleBackup.destroy();
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
