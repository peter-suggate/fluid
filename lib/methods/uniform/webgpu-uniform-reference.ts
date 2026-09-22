import { UniformScratchArena } from "./uniform-scratch-arena";
import { uniformPageHasNativeCoordinates, uniformPageHasRectangularCoverage } from "./uniform-page-execution";
import { UniformPageDomainPublication } from "./uniform-page-domain-publication";
import { UniformTexturePages } from "./uniform-texture-pages";
import {initialUniformPageDomain,type UniformPageDomain} from "./uniform-page-domain";
import { UNIFORM_VOLUME_PAGE_ENTRIES, type UniformVolumePageShaderOptions } from "./uniform-volume-pages.wgsl";
import { UniformSurfaceVolumeCorrection } from "./webgpu-uniform-surface-volume";
import { uniformDensityPostProcessingEnabled } from "./uniform-options";
export { uniformDensityPostProcessingEnabled } from "./uniform-options";
import type { DenseLevelSetVolumeConsumerSource } from "../../core/levelset-consumer-abi";
import {
  SOLVE_WINDOW_HOST_GROUPS_WORD, SOLVE_WINDOW_RECORD_WORDS,
  type GPUFluidSolveWindowSource, type GPUFluidTileClassSource, type GPUFluidVolumePageSource,
} from "../../core/method-view-records";
import { UNIFORM_VOLUME_PHASE } from "./uniform-volume-stages";
import {
  UNIFORM_VOLUME_EDGE_BYTES,
  UNIFORM_VOLUME_ENTRIES,
  UNIFORM_VOLUME_SHARPEN_ENTRIES,
  UNIFORM_VOLUME_SHARPEN_TILE_COUNT_WORD,
  UNIFORM_VOLUME_SHARPEN_TILE_MAP_WORD,
  UNIFORM_VOLUME_TILE_CLASSIFY_ENTRY,
  UNIFORM_VOLUME_TILE_WORK_OVERRIDE,
  UNIFORM_VOLUME_TWO_LEVEL_COUNTER_WORDS,
  UNIFORM_VOLUME_TWO_LEVEL_ENTRIES,
  UNIFORM_VOLUME_TWO_LEVEL_WORDS_PER_TILE,
} from "./uniform-volume.wgsl";
import { createUniformReferenceComputeShader } from "./webgpu-uniform-reference.wgsl";
import { uniformAbOn } from "./uniform-ab-switch";
import { uniformVolumeInitialPhi, uniformInitialVolume } from "./uniform-volume-initial";
import { averageInflowStrength, createInflowGridBoundary, type InflowGridBoundary } from "../../core/inflow-boundary";
import type { SceneDescription } from "../../core/model";
import { planUniformHostAllocation } from "./uniform-host-allocation";
import { initializeRigidBodies, type RigidBodyState } from "../../core/rigid-body";
import { sceneLatticeDimensions } from "../../core/scene-lattice";
import { planGPUAdvance } from "../../core/tall-cell-diagnostics";
import type { GPUQuality } from "../../core/gpu-quality";
import { sceneHasTerrain } from "../../core/terrain";
import {
  GPU_RIGID_EXCHANGE_BYTES,
  type GPUEulerianInfo,
  type GPURigidLoad,
  type GPUVelocityTransport,
} from "../../core/webgpu-eulerian";
import { GPUInitializationTaskRunner, type GPUInitializationTask } from "../../core/gpu-initialization";
import type {
  GPUSolverInstance,
  GPUInitializationReporter,
  InjectedLiquidBall,
  MethodParamValues,
} from "../../core/method-contract";
import { WebGPURigidBodySystem } from "../../core/webgpu-rigid-body";
import { uniformReferenceComputeShader } from "./webgpu-uniform-reference.wgsl";
import { WebGPUUniformVelocityExtrapolator } from "./webgpu-uniform-velocity-extrapolation";
import {
  UNIFORM_CM11A_DEFAULT_BUDGET_HEADROOM,
  UNIFORM_CM11A_FULL_CYCLES,
  UNIFORM_CM11A_MINIMUM_CYCLE_BUDGET,
  UNIFORM_CM11A_POST_SWEEPS,
  UNIFORM_CM11A_PRE_SWEEPS,
  UNIFORM_CM11A_V_CYCLES,
  uniformCM11aCycleBudget,
  WebGPUUniformPressureMultigrid,
  planUniformCM11aWindow,
  seatUniformCM11aWindow,
  type UniformCM11aSchedule,
} from "./webgpu-uniform-pressure-multigrid";
import type { UniformCM11aCoarsestCapture, UniformCM11aPlanStage } from "./webgpu-uniform-pressure-multigrid";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import {
  CPUPerformanceTrace,
  GPUQueueWallPerformanceTraceRecorder,
  GPUStageTimestampRecorder,
  type GPUTimestampPhase,
} from "../../core/performance-trace";
import { usePerformanceInstrumentationStore } from "../../core/stores/performance-instrumentation-store";
import { gpuPhysicsPerformanceActivityFrameId } from "../../core/gpu-performance-activity";
import { UNIFORM_ADVANCE_PHASE } from "./uniform-stages";
export { UNIFORM_ADVANCE_PHASE } from "./uniform-stages";
export { UNIFORM_FLUID_PIPELINE } from "./uniform-pipeline";
import { UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS, UNIFORM_GAMMA_DIFFUSION_MAX_ITERATIONS } from "./parameters";
export { UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS, UNIFORM_GAMMA_DIFFUSION_MAX_ITERATIONS } from "./parameters";
import { UNIFORM_PAPER_DT_S, uniformPaperAdvanceReady } from "./uniform-paper";
import { liveFluidEditRefusal, type LiveFluidEdit, type LiveFluidEditResult } from "../../core/live-fluid-edit";
import { SolidOccupancyMask } from "../../core/solid-occupancy-mask";
import { solidWorldForScene } from "../../core/solid-world";

export { UNIFORM_PAPER_DT_S } from "./uniform-paper";

export interface WebGPUUniformReferenceOptions {
  /** Opt-in paged storage for transport/sharpening scratch (3D only). */
  volumePages?: 16 | 32 | "auto";
  /** Production Uniform Geometric domain; false exists only for the numerical oracle. */
  pageDomain?: boolean;
  /** QA backing oracle; complete rectangular production residency uses native fields. */
  fieldStorageForQA?: "dense" | "paged";
  /** Full-domain oracle for the separately compiled vertex work window. */
  phiWindowForQA?: false;
  /** Full-lattice pressure smoothing control for the work-list regression. */
  pressureSmoothingForQA?: "dense";
  /** Former field atlas and vertex traversal retained for identical-input QA. */
  phiStorageForQA?: "paged";
  phiReadAuditForQA?: boolean;
  phiLiteralLoopsForQA?: boolean;
  /** QA-only former pressure layouts; production uses native execution fields. */
  pressureStorageForQA?: "paged" | "paged-logical";
  /** Internal scheduling oracle; paged work is the production default. */
  volumePageWork?: boolean;
  /** Scene-parity oracle only: one symmetry-depth cell, with a 2D pressure hierarchy. */
  referenceDimension?: 2 | 3;
  /** Independent dense vertex level set and conservative cell volume. */
  geometricVolume?: boolean;
  geometricRedistance?: boolean;
  /** Separate backing is the exact numerical/performance control. */
  scratchStorageForQA?: "separate";
  /** Retain stage fields, including inactive cells, for full-field diagnostics. */
  retainStageDiagnosticsForQA?: boolean;
  /** Global surface-volume constraint; defaults on for 3D geometric volume. */
  totalSurfaceVolume?: boolean;
  surfaceDeficitBalancing?: boolean;
  /**
   * 4h work map for the geometric sharpening sweeps, on by default with
   * `geometricVolume`. False is the dense control: identical numerics, every
   * tile visited. Both variants are always resident, so `applyRuntimeValues`
   * can flip this between steps.
   */
  geometricTileWork?: boolean;
  /** Sec. 3.3 FIM sweep budget; clamped to the extrapolator's wavefront ceiling. */
  extensionFrontSweeps?: number;
  /** Nearest original-source hierarchy; defaults on for geometric volume. */
  sourceAwareExtension?: boolean;
  /** Diagnostic control for checking the fused final transfer. */
  fuseExtensionPack?: boolean;
  /**
   * Discard |V| below this many cell volumes wherever Sec. 3.4 or Sec. 3.5
   * finally writes V, tiny negatives included. Zero is off and stores the
   * untreated sum bit for bit.
   */
  volumeDustThreshold?: number;
  /**
   * Experiment E1: sample velocity from the restricted 4h level outside the
   * fine tile map. Numerics only -- every lattice and every dispatch is the
   * size it was; nothing shrinks yet.
   */
  twoLevelVelocity?: boolean;
  /** Chebyshev dilation of the E1 seed tiles, in 4h tiles. */
  twoLevelFineReach?: number;
  /**
   * E2: run the velocity extension's finest passes on the shell tiles instead
   * of densely. Meaningful only while `twoLevelVelocity` is on, because the
   * shrunk fine field is exactly what the 4h sampler replaces.
   */
  twoLevelExtensionTiles?: boolean;
  /** Extra 4h tiles the shell adds past the fine set; absent derives it. */
  twoLevelShellReach?: number;
  /**
   * E2b: velocity advection and the projection take their far-air arm directly
   * in cells outside the fine tiles. Also gated on `twoLevelVelocity`, since the
   * tile map is only built when that is on.
   */
  twoLevelAdvectionTiles?: boolean;
  /**
   * E3: run every pass of the conservative volume transport only on the live
   * tile set. Gated on `twoLevelVelocity` (which builds the class map) and on a
   * positive `volumeDustThreshold`, which is what makes "V is zero outside the
   * set" -- the predicate that keeps the restriction lossless -- true.
   */
  transportTiles?: boolean;
  /** Extra 4h tiles the transport set adds past the fine set. */
  transportReach?: number;
  /**
   * Geometric only. A cell holding at least half its open capacity in V owns a
   * pressure row even where centre phi is positive, so sub-half-cell films keep
   * incompressibility and Sec. 3.7's excess divergence can reach stacked V.
   * "abandoned" (or true) restricts that to cells with no phi-liquid face
   * neighbour; "all" lets V move the free surface beside phi-liquid too, which
   * roughens every surface (docs/research/uniform-geometric-thin-film-2026-09-19).
   */
  volumePressureRows?: boolean | "off" | "abandoned" | "all";
  /**
   * Geometric only, all off by default
   * (docs/uniform-geometric-phi-volume-agreement-handoff.md). Compaction lets
   * sharpening pour V toward deeper liquid at any depth so voids inside the
   * liquid refill; the seed writes V into phi where phi has no surface at all;
   * the shift moves band phi along its normal by a slow, tent-gathered V - fill
   * residual. Gain 0 disables the shift; the clamp is in cells a step.
   */
  volumeCompaction?: boolean;
  phiSeedFromVolume?: boolean;
  phiAgreementGain?: number;
  phiAgreementClamp?: number;
  /** GPU-resident sparse work boxes; false retains the original dense control. */
  activeRegion?: boolean;
  /**
   * Plan the CM11a pressure hierarchy on the solve window instead of the
   * domain. Only effective while `activeRegion` is on; defaults to on there.
   */
  pressureWindow?: boolean;
  /** Velocity transport used by Algorithm 1 step 3. */
  velocityTransport?: GPUVelocityTransport;
  /** Reject hierarchy-only air samples when updating liquid momentum. */
  liquidOnlyVelocityAdvection?: boolean;
  /** Diagnostic-only switch; the authored paper method always enables Sec. 3.5. */
  densitySharpening?: boolean;
  /** Return the density removed by Sec. 3.5 through local Algorithm 2 scatter. */
  sharpeningMassCorrection?: boolean;
  /** Ordered six-pass Sec. 3.4 iterations; zero cleanly bypasses the stage. */
  gammaDiffusionIterations?: number;
  /** Multiplier over the paper's 3dt sharpening pseudo-time. */
  sharpeningStrength?: number;
  /** Algorithm 2 maximum gradient-trace distance, in cells. */
  sharpeningDistance?: number;
  /** Sec. 3.6 cut-cell excess redistribution. */
  solidExcessCorrection?: boolean;
  /** Two-way fluid/body exchange and rigid integration. */
  rigidCoupling?: boolean;
  /** CM11a cycle and smoothing schedule. */
  pressureSchedule?: UniformCM11aSchedule;
  /**
   * "lagged" encodes only as many cycles as the last observed step needed;
   * "fixed" encodes the whole configured schedule, which is what the solver
   * always did. Runtime-switchable.
   */
  pressureCycleBudget?: "lagged" | "fixed";
  /** Test-only launch-shape control; production selects once from page geometry. */
  pressureCycleDispatch?: "direct" | "indirect";
  /** Cycles the lagged budget adds above the last observed demand. */
  pressureBudgetHeadroom?: number;
  /** Paper Sec. 3.8 render reconstruction; Results states it is normally off. */
  densityPostProcessing?: boolean;
  /**
   * "paper" advances at the paper's simulation step (1/30 s, the value used
   * by every example in Sec. 4); "scene" honors the scene-authored maxDt.
   * The method's advection/sharpening balance is only calibrated at the
   * paper's step regime: far below it, per-resample transport diffusion
   * outruns Sec. 3.5 sharpening and the interface dilutes below the 0.5
   * isovalue, leaving dynamically inert mass hanging in mid-air.
   */
  timeStep?: "paper" | "scene";
  deferPipelineCompilation?: boolean;
}

// Sec. 3.4 permits one through seven gamma-diffusion repetitions per time
// step. One repetition is the reference schedule: each dimensional sweep is
// one Jacobi update from a single snapshot, as specified by LAF11/CM12. More
// repetitions remain available explicitly, but are not a neutral robustness
// setting: they apply more physical/numerical diffusion per simulation step.


interface UniformReferencePipelines {
  scanActiveRegion: GPUComputePipeline;
  scanExternalActiveSources: GPUComputePipeline;
  reduceActiveRegionSummaries: GPUComputePipeline;
  reduceExternalActiveRegionSummaries: GPUComputePipeline;
  finalizeActiveRegion: GPUComputePipeline;
  semiLagrangian: GPUComputePipeline;
  advect: GPUComputePipeline;
  reverse: GPUComputePipeline;
  correct: GPUComputePipeline;
  project: GPUComputePipeline;
  coupleRigid: GPUComputePipeline;
  reduce: GPUComputePipeline;
  extrapolationAuthority: GPUComputePipeline;
  extrapolationAuthorityDense: GPUComputePipeline;
  extrapolationDensityAuthority: GPUComputePipeline;
  traceGammaBeta: GPUComputePipeline;
  scatterDensityDeficit: GPUComputePipeline;
  gatherDensity: GPUComputePipeline;
  diffuseGammaX: GPUComputePipeline;
  diffuseGammaY: GPUComputePipeline;
  diffuseGammaZ: GPUComputePipeline;
  postprocessBlurX: GPUComputePipeline;
  postprocessBlurY: GPUComputePipeline;
  postprocessBlurZ: GPUComputePipeline;
  postprocessResolve: GPUComputePipeline;
  wallFilmResolve: GPUComputePipeline;
  sharpenCompute: GPUComputePipeline;
  sharpenScatter: GPUComputePipeline;
  sharpenResolve: GPUComputePipeline;
  scatterSolidExcess: GPUComputePipeline;
  resolveSolidExcess: GPUComputePipeline;
}

const PIPELINES = [
  ["scanActiveRegion", "Scan active liquid bounds", "scanActiveRegion", false],
  ["scanExternalActiveSources", "Scan external active sources", "scanExternalActiveSources", false],
  ["reduceActiveRegionSummaries", "Reduce active liquid summaries", "reduceActiveRegionSummaries", false],
  ["reduceExternalActiveRegionSummaries", "Reduce external-source summaries", "reduceExternalActiveRegionSummaries", false],
  ["finalizeActiveRegion", "Finalize active liquid dispatches", "finalizeActiveRegion", false],
  ["semiLagrangian", "Semi-Lagrangian velocity advection and body forces", "semiLagrangianAdvection", false],
  ["advect", "Bounded MacCormack velocity prediction", "advect", false],
  ["reverse", "Bounded MacCormack reverse advection", "reverseAdvection", false],
  ["correct", "Bounded MacCormack correction and body forces", "correctAdvection", false],
  ["project", "Project velocity", "project", false],
  ["coupleRigid", "Couple rigid bodies", "coupleRigid", false],
  ["reduce", "Reduce diagnostics", "reduceDiagnostics", false],
  ["extrapolationAuthority", "Build Sec. 3.3 interface authority", "buildExtrapolationAuthority", false],
  ["extrapolationAuthorityDense", "Seed dense Sec. 3.3 interface authority", "buildDenseExtrapolationAuthority", false],
  ["extrapolationDensityAuthority", "Build Sec. 3.3 rho-prime authority", "buildExtrapolationDensityAuthority", false],
  ["traceGammaBeta", "Trace gamma and scatter beta", "traceGammaAndBeta", false],
  ["scatterDensityDeficit", "Scatter density and gamma deficits", "scatterDensityDeficit", false],
  ["gatherDensity", "Gather conservative surface density", "gatherConservativeDensity", false],
  ["diffuseGammaX", "Diffuse gamma along x (Jacobi)", "diffuseGammaX", false],
  ["diffuseGammaY", "Diffuse gamma along y (Jacobi)", "diffuseGammaY", false],
  ["diffuseGammaZ", "Diffuse gamma along z (Jacobi)", "diffuseGammaZ", false],
  ["postprocessBlurX", "Post-process gamma blur x", "postprocessBlurX", false],
  ["postprocessBlurY", "Post-process gamma blur y", "postprocessBlurY", false],
  ["postprocessBlurZ", "Post-process gamma blur z", "postprocessBlurZ", false],
  ["postprocessResolve", "Resolve sub-grid surface density", "postprocessResolve", false],
  ["wallFilmResolve", "Resolve mass-proportional wall films", "wallFilmResolve", false],
  ["sharpenCompute", "Compute interface sharpening", "sharpenCompute", false],
  ["sharpenScatter", "Scatter conserved interface mass", "sharpenScatter", false],
  ["sharpenResolve", "Resolve conserved interface mass", "sharpenResolve", false],
  ["scatterSolidExcess", "Scatter partial-solid density excess", "scatterSolidExcess", false],
  ["resolveSolidExcess", "Resolve partial-solid density excess", "resolveSolidExcess", false],
] as const;

/**
 * Boundaries are free (`GPUStageTimestampRecorder` splices them into passes the
 * frame already encodes) but the trace's query set, resolve and map are not.
 */
const UNIFORM_PHYSICS_TRACE_CADENCE_MS = 100;

/** Active-region buffer ABI shared by the main kernels and pressure hierarchy. */
const UNIFORM_ACTIVE_MAIN_DISPATCH_OFFSET = 13 * 4;
const UNIFORM_ACTIVE_LEVEL_WORDS = 10;
const UNIFORM_ACTIVE_LEVEL_BASE_WORD = 16;
const UNIFORM_ACTIVE_MAX_LEVELS = 16;
/**
 * Uniform Geometric's phi passes run on the (n+1)^3 vertex lattice, which is
 * one vertex wider than the window's cell box on every axis and shares its
 * origin. Its indirect record sits above the level table; the fourth word pads
 * the header to a four-word stride, and `ACTIVE_SUMMARY_BASE` in the shader is
 * this word plus four.
 */
const UNIFORM_ACTIVE_VERTEX_DISPATCH_WORD = UNIFORM_ACTIVE_LEVEL_BASE_WORD
  + UNIFORM_ACTIVE_MAX_LEVELS * UNIFORM_ACTIVE_LEVEL_WORDS;
const UNIFORM_ACTIVE_VERTEX_DISPATCH_OFFSET = UNIFORM_ACTIVE_VERTEX_DISPATCH_WORD * 4;
/**
 * Group counts the HOST chose for this step's direct dispatches, and the
 * violation ledger `finalizeActiveRegion` keeps against them. An indirect
 * launch costs 15-25 us on this Dawn/Metal lane against 3-6 us for a direct
 * one, and the window converts about a thousand launches a step, so the counts
 * are chosen on the CPU from a lagged box while the ORIGIN every kernel reads
 * stays the GPU's exact one.
 */
const UNIFORM_ACTIVE_CPU_LEVEL_BASE_WORD = UNIFORM_ACTIVE_VERTEX_DISPATCH_WORD + 4;
const UNIFORM_ACTIVE_CPU_MAIN_WORD = UNIFORM_ACTIVE_CPU_LEVEL_BASE_WORD
  + UNIFORM_ACTIVE_MAX_LEVELS * 3;
const UNIFORM_ACTIVE_CPU_VERTEX_WORD = UNIFORM_ACTIVE_CPU_MAIN_WORD + 3;
const UNIFORM_ACTIVE_CPU_MODE_WORD = UNIFORM_ACTIVE_CPU_VERTEX_WORD + 3;
const UNIFORM_ACTIVE_VIOLATION_WORD = UNIFORM_ACTIVE_CPU_MODE_WORD + 1;
const UNIFORM_ACTIVE_VIOLATION_AXES_WORD = UNIFORM_ACTIVE_VIOLATION_WORD + 1;
/**
 * Per-axis travel in cells, both directions summed, packed three to a word
 * (ten bits each) by `finalizeActiveRegion`. This is how much wider the exact
 * box can get in one step, per axis, and it is what the lagged group counts
 * below are padded with. The two words under it are the finalize's own
 * scratch for the plus and minus halves.
 */
const UNIFORM_ACTIVE_TRAVEL_TOTAL_WORD = UNIFORM_ACTIVE_VIOLATION_AXES_WORD + 3;
/** Origin of the window-local CM11a lattice, in simulation cells. */
const UNIFORM_ACTIVE_PRESSURE_ORIGIN_WORD = UNIFORM_ACTIVE_TRAVEL_TOTAL_WORD + 1;
/** Mirror of VERTEX_PHI_REACH: the phi stage's own read reach, in vertices. */
const UNIFORM_VERTEX_PHI_REACH = 6;
const UNIFORM_ACTIVE_HEADER_WORDS = 256;
// The solve-window view decodes this header by the words `method-view-records`
// names: the seed box at 0, the window at 7, and the host's main group counts.
if (UNIFORM_ACTIVE_CPU_MAIN_WORD !== SOLVE_WINDOW_HOST_GROUPS_WORD
  || UNIFORM_ACTIVE_HEADER_WORDS < SOLVE_WINDOW_RECORD_WORDS) {
  throw new Error("The active-region header moved; update SOLVE_WINDOW_* in core/method-view-records.ts");
}
const UNIFORM_ACTIVE_SUMMARY_BYTES = 48;
/**
 * Steps of lag the host assumes between the box the GPU computed and the box
 * its group counts must still cover. The never-awaited readback is one step
 * behind at best; the measured maximum over a sixty-frame capture was also
 * one, so two is that plus a skipped copy. The containment flag and the
 * whole-domain fallback are the safety net, not this number.
 *
 * It is a LAUNCH SIZE only: every windowed kernel exits the threads whose
 * workgroup overruns the published window, so raising or lowering this changes
 * how many threads return immediately and nothing about the answer.
 */
const UNIFORM_WINDOW_LAG_STEPS = 2;
/** Steps of whole-domain counts a clipped step buys. */
const UNIFORM_WINDOW_VIOLATION_PENALTY_STEPS = 8;
/**
 * Cells of slack on the CPU-known starting wet box, for the steps before the
 * first readback lands. Generous on purpose: a lattice one alignment step too
 * wide costs a little memory, and a lattice too small costs a re-plan.
 */
const UNIFORM_PRESSURE_WINDOW_SEED_PAD = 32;
/**
 * Consecutive steps a smaller capacity must suffice for before the lattice
 * shrinks to it. Growth is immediate; only shrinking waits, because a surface
 * breathing across an alignment boundary would otherwise re-plan forever.
 */
const UNIFORM_PRESSURE_WINDOW_SHRINK_STEPS = 30;
/** CM11a instances kept alive, including the domain-capacity one. */
const UNIFORM_PRESSURE_WINDOW_CACHE = 3;
/**
 * Cells of headroom at which the next capacity starts being built.
 *
 * Growth is the one re-plan that cannot wait: the step that outgrows the
 * capacity needs the bigger lattice in the same step. Building it costs ~10 ms
 * of host walk, which at the budget below is several steps, so the trigger has
 * to lead the box by several steps of ITS OWN travel and not by a fixed
 * distance -- eight cells of headroom is five steps of a calm surface and one
 * step of a far-wall impact, which is exactly when the hitch would land.
 */
const UNIFORM_PRESSURE_PREWARM_HEADROOM = 8;
/** Steps of the box's own measured travel the prewarm tries to lead by. */
const UNIFORM_PRESSURE_PREWARM_STEPS = 6;
/** Host milliseconds a step may spend on the prewarm's plan walk. */
const UNIFORM_PRESSURE_PREWARM_BUDGET_MS = 2;

/** The multigrid's self-reported schedule groups, mapped onto the phase table. */
const UNIFORM_PRESSURE_STAGE_PHASE: Readonly<Record<UniformCM11aPlanStage, GPUTimestampPhase>> = Object.freeze({
  setup: UNIFORM_ADVANCE_PHASE.pressureSetup,
  "full-cycle": UNIFORM_ADVANCE_PHASE.pressureFullCycles,
  "v-cycle": UNIFORM_ADVANCE_PHASE.pressureVCycles,
  finish: UNIFORM_ADVANCE_PHASE.pressureFinish,
});

/**
 * Uniform numerical implementation with page-backed geometric fields and a
 * separate dense paper/reference path.
 *
 * This class deliberately has no octree option or adaptive compatibility
 * branch. Its allocations, pipelines, step graph, and diagnostics are owned
 * entirely by the `uniform` method plugin.
 */
export class WebGPUUniformReferenceSolver implements GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  private readonly geometricVolume: boolean;
  /** Live toggle; `sharpenTileWork` is what an encode actually honours. */
  private geometricTileWork: boolean;
  /** 4h records the third conditioning plane can hold; 0 disables the map. */
  private readonly sharpenTileCount: number;
  private sharpenTilePipelines: Partial<Record<typeof UNIFORM_VOLUME_SHARPEN_ENTRIES[number], GPUComputePipeline>> = {};
  private tileClassifyPipeline?: GPUComputePipeline;
  /** The classify dispatch ran in the most recent encoded step. */
  private sharpenTileMapEncoded = false;
  private geometricRedistance: boolean;
  /** Sec. 3.4/3.5 rounding-residue floor in cell volumes; 0 is off. */
  private volumeDustThreshold: number;
  /** Experiment E1 live toggle and its Chebyshev fine reach, in 4h tiles. */
  private twoLevelVelocity: boolean;
  private twoLevelFineReach: number;
  /** E2: the extension's finest passes run on the shell tiles, not densely. */
  private twoLevelExtensionTiles: boolean;
  /** E2b: advection and projection take their far-air arm outside the fine tiles. */
  private twoLevelAdvectionTiles: boolean;
  /** How far past the fine set the extension must still be exact, in 4h tiles. */
  private twoLevelShellReach: number;
  /** E3: the twelve transport passes run on the live tile set, not densely. */
  private transportTiles: boolean;
  /** Extra 4h tiles the transport set adds past the fine set. */
  private transportReach: number;
  /** V may claim a pressure row that centre phi alone would deny. */
  private volumePressureRows: 0 | 1 | 2;
  /** phi/V agreement stages; see the option docs. */
  private volumeCompaction: boolean;
  private phiSeedFromVolume: boolean;
  private totalSurfaceVolume: boolean;
  private surfaceDeficitBalancing: boolean;
  private readonly surfaceDeficitBalanceBytes: number;
  private surfaceVolumeCorrection?: UniformSurfaceVolumeCorrection;
  private phiAgreementGain: number;
  private phiAgreementClamp: number;
  /** The transport restriction was encoded in the most recent step. */
  private transportTilesEncoded = false;
  /** Coarse cells whose E1 tables fit the conditioning plane; 0 disables E1. */
  private twoLevelTileCount: number;
  private twoLevelPipelines: Partial<Record<typeof UNIFORM_VOLUME_TWO_LEVEL_ENTRIES[number], GPUComputePipeline>> = {};
  /** The E1 map was built in the most recent encoded step. */
  private twoLevelEncoded = false;
  /** The E1 records, as the fine-tiles view binds them; see `tileClassSource`. */
  private readonly tileClassRecords?: GPUFluidTileClassSource;
  private readonly vertexPhiField?: GPUTexture;
  get vertexPhiTexture(): GPUTexture | undefined { return this.vertexPhiField && this.present(this.vertexPhiField); }
  private readonly fieldPages?: UniformTexturePages;
  private readonly scratchArena?: UniformScratchArena;
  private present(field: GPUTexture): GPUTexture { return this.fieldPages?.publication(field) ?? field; }
  readonly denseLevelSetVolumeSource?: DenseLevelSetVolumeConsumerSource;
  private readonly vertexPhiScratch?: GPUTexture;
  /** Most recent transported phi, before closest-point redistancing (diagnostics). */
  get advectedVertexPhiTexture(): GPUTexture | undefined { return this.vertexPhiScratch && this.present(this.vertexPhiScratch); }
  private volumeEdges?: GPUBuffer;
  private readonly pageDomain?: UniformPageDomain;
  private readonly nativePageCoordinates: boolean = false;
  private readonly nativeRootExecution: boolean = false;
  private readonly pageDomainPublication?: UniformPageDomainPublication;
  private readonly pageDomainDispatch?: GPUBuffer;
  private readonly pageDomainView?: GPUBuffer;
  private readonly volumeTransportPageView?: GPUBuffer;
  private readonly volumePageEdge: 0 | 16 | 32;
  private readonly volumePageConfig?: UniformVolumePageShaderOptions;
  private readonly volumeWorkDispatch?: GPUBuffer;
  private readonly volumeWorkCounts?: GPUBuffer;
  private readonly volumePageSharpenFlag?: GPUBuffer;
  private pagePipelines: Partial<Record<typeof UNIFORM_VOLUME_PAGE_ENTRIES[number], GPUComputePipeline>> = {};
  private readonly volumeDonorSums?: GPUBuffer;
  private volumePipelines: Partial<Record<typeof UNIFORM_VOLUME_ENTRIES[number], GPUComputePipeline>> = {};
  private phiReverseGroup?: GPUBindGroup;
  private readonly shaderSource: string;
  private readonly phiShaderSource?: string;
  private readonly phiRedistanceShaderSource?: string;
  private readonly phiRegion?: GPUBuffer;
  private readonly phiDispatch?: GPUBuffer;
  private readonly groupDescriptors = new WeakMap<GPUBindGroup, GPUBindGroupDescriptor>();
  private readonly phiGroups = new Map<GPUBindGroup, GPUBindGroup>();
  private readonly pressureInputLayout: GPUBindGroupLayout;
  readonly volumeTexture: GPUTexture;
  get surfaceFieldTexture(): GPUTexture {
    return this.present(this.surfaceB);
  }
  readonly columnBaseTexture: GPUTexture;
  readonly velocityTexture: GPUTexture;
  /** Velocity after advection/forces and before the pressure projection. */
  get preProjectionVelocityTexture(): GPUTexture { return this.present(this.velocityB); }
  /** Padded float32 velocity extension retained for opt-in comparison diagnostics. */
  readonly extrapolatedVelocityTexture: GPUTexture;
  /** FIM xyz distances plus the 3-bit active mask in w, for Dawn conformance diagnostics. */
  get extrapolationActiveStateTexture(): GPUTexture {
    this.enableStageDiagnosticsForQA();
    return this.present(this.velocityExtrapolator.activeStateTexture);
  }
  private stageDiagnosticsRequested = false;
  private readonly retainStageDiagnosticsForQA: boolean;
  /** Full FIM diagnostics must be retained when constructing the solver. */
  enableStageDiagnosticsForQA(): void {
    if(this.stageDiagnosticsRequested || !this.scratchArena)return;
    if(!this.retainStageDiagnosticsForQA)throw new Error("Construct the solver with retainStageDiagnosticsForQA to inspect intermediate fields");
    this.stageDiagnosticsRequested=true;
  }
  readonly extrapolationActiveFrontPassCeiling: number;
  private readonly symmetryStageAuditFields?: WebGPUUniformReferenceSolver["symmetryStageAuditTextures"];
  readonly symmetryStageAuditTextures?: Readonly<{
    preExtrapolationVelocity: GPUTexture;
    previousRawDensity: GPUTexture;
    extrapolationDensityAuthority: GPUTexture;
    densityAdvection: GPUTexture;
    densityDiffusion: GPUTexture;
    densitySharpening: GPUTexture;
    gammaPostAdvection: GPUTexture;
    gammaPostDiffusion: GPUTexture;
    velocityPrediction: GPUTexture;
    predictedExtrapolation: GPUTexture;
    reverseAdvection: GPUTexture;
    velocityAdvection: GPUTexture;
    pressureProjection: GPUTexture;
  }>;
  /** Negative domain MAC faces paired with preExtrapolationVelocity. */
  readonly symmetryStageAuditNegativeBoundaryVelocity?: GPUBuffer;
  /** Scalar-packed x/y/z negative-face plane byte length. */
  readonly negativeBoundaryVelocityBytes: number;
  /** Negative x/y/z domain MAC faces paired with velocityTexture. */
  get negativeBoundaryVelocityBuffer(): GPUBuffer { return this.boundaryVelocityA; }
  /**
   * Read-only accepted pressure/gamma for matched-lattice Dawn comparisons.
   *
   * With the pressure lattice planned on the window the texture is
   * window-local: it has `latticeDimensions` haloed cells and its cell
   * (i,j,k) is simulation cell (i,j,k) - 1 + `latticeOrigin`. A caller that
   * assumes the domain plus a one-cell halo must apply that or keep the
   * lattice off.
   */
  get gridVelocityBoundary(): GPUBufferBinding { return { buffer: this.boundaryVelocityA }; }
  get gridPressureTexture(): GPUTexture { return this.pressureMultigrid.pressureTexture; }
  get gridPressureOrigin(): readonly [number, number, number] { return this.pressureWindowOrigin; }
  get physicsFieldsForQA() {
    return { pressure: this.pressureMultigrid.pressureTexture, gamma: this.present(this.gammaA),
      latticeOrigin: [...this.pressureWindowOrigin] as [number, number, number],
      latticeDimensions: this.pressureWindowCapacity.map((value) => value + 2) as
        [number, number, number] };
  }
  /** Eight vec4 decision records for every stored MAC face/component. */
  readonly symmetryStageAuditMacCormackBuffer?: GPUBuffer;
  /** Fixed-point beta produced by Sec. 3.4 before deficit scattering. */
  readonly symmetryStageAuditBetaBuffer?: GPUBuffer;

  private readonly velocityA: GPUTexture;
  private readonly velocityB: GPUTexture;
  private readonly velocityC: GPUTexture;
  private readonly velocityD: GPUTexture;
  private readonly boundaryVelocityA: GPUBuffer;
  private readonly boundaryVelocityB: GPUBuffer;
  private readonly boundaryVelocityC: GPUBuffer;
  private readonly boundaryVelocityD: GPUBuffer;
  private readonly pressureA: GPUTexture;
  private readonly pressureB: GPUTexture;
  private readonly volumeA: GPUTexture;
  private readonly volumeB: GPUTexture;
  private readonly surfaceA: GPUTexture;
  private readonly surfaceB: GPUTexture;
  private readonly gammaA: GPUTexture;
  private readonly gammaB: GPUTexture;
  private readonly heightA: GPUTexture;
  private readonly heightB: GPUTexture;
  private readonly terrainTexture: GPUTexture;
  private readonly transportA: GPUTexture;
  private readonly transportB: GPUTexture;
  private readonly velocityExtrapolator: WebGPUUniformVelocityExtrapolator;
  private pressureMultigrid: WebGPUUniformPressureMultigrid;
  /**
   * CM11a instances by capacity, most recently used last.
   *
   * A window-local lattice is planned for a CAPACITY, not for the domain, and
   * the capacity changes only when the liquid's box outgrows it (at once) or
   * has been comfortably inside a smaller one for a while. Keeping the last
   * few alive means the common sloshing case never builds anything: the
   * instances share their compiled pipelines and bind-group layouts, so all a
   * new one costs is textures, param buffers, bind groups and a plan.
   */
  private readonly pressureInstances = new Map<string, WebGPUUniformPressureMultigrid>();
  private readonly pressureDomainKey: string;
  private pressureWindowLattice = false;
  private pressureWindowLatticeActive = false;
  private pressureWindowOrigin: [number, number, number] = [0, 0, 0];
  private pressureWindowCapacity: [number, number, number];
  private pressureWindowDomainSteps = 0;
  private pressureWindowShrinkStreak = 0;
  private pressureWindowReplans = 0;
  private pressureWindowReplanMs = 0;
  /** The capacity being built ahead of the window needing it. */
  private pressurePrewarm?: { key: string; instance: WebGPUUniformPressureMultigrid };
  private pressurePrewarmMs = 0;
  private pressurePrewarmSteps = 0;
  // Origin(3), capacity(3), mode(1).
  private pressureWindowHeaderWords = new Uint32Array(7);
  /** The starting wet box, for the steps before the first readback lands. */
  private pressureWindowSeedBox?: { minimum: number[]; maximum: number[] };
  /** Coarsest-capture request, re-armed on whichever instance is current. */
  private pressureCoarsestCapture?: number;
  private readonly params: GPUBuffer;
  private readonly solidVoxelScratchOffsetWords: number;
  private readonly reductions: GPUBuffer;
  private readonly conditioningScratch: GPUBuffer;
  private readonly activeRegion: GPUBuffer;
  private readonly activeScratch: GPUBuffer;
  private readonly activeDispatch: GPUBuffer;
  /** Distinct storage binding for the compile-time-disabled MacCormack audit.
   * WebGPU still validates writable binding aliasing even when the shader's
   * constant-false branch cannot execute. */
  private readonly macCormackAuditBinding: GPUBuffer;
  private readonly rigidExchange: GPUBuffer;
  private readonly rigidSystem: WebGPURigidBodySystem;
  private readonly mainLayout: GPUBindGroupLayout;
  private readonly mainPipelineLayout: GPUPipelineLayout;
  private readonly extrapolationAuthorityGroup: GPUBindGroup;
  private readonly semiLagrangianGroup: GPUBindGroup;
  private readonly advectGroup: GPUBindGroup;
  private readonly reverseGroup: GPUBindGroup;
  private readonly correctGroup: GPUBindGroup;
  private readonly pressureMultigridGroup: GPUBindGroup;
  private projectGroup: GPUBindGroup;
  private readonly makeProjectGroup: (pressure: GPUTexture) => GPUBindGroup;
  private readonly rigidGroup: GPUBindGroup;
  private readonly reductionGroup: GPUBindGroup;
  private readonly densityTraceGroup: GPUBindGroup;
  private readonly volumeDonorGroup: GPUBindGroup;
  private readonly densityScatterGroup: GPUBindGroup;
  private readonly densityGatherGroup: GPUBindGroup;
  private readonly gammaDiffusionGroups: readonly [GPUBindGroup, GPUBindGroup];
  private readonly postprocessBlurXGroup: GPUBindGroup;
  private readonly postprocessBlurYGroup: GPUBindGroup;
  private readonly postprocessBlurZGroup: GPUBindGroup;
  private readonly postprocessResolveGroup: GPUBindGroup;
  private readonly wallFilmResolveGroup: GPUBindGroup;
  private readonly sharpenComputeGroup: GPUBindGroup;
  private readonly sharpenScatterGroup: GPUBindGroup;
  private readonly sharpenResolveGroup: GPUBindGroup;
  private readonly solidEntryScatterGroup: GPUBindGroup;
  private readonly solidEntryResolveGroup: GPUBindGroup;
  private readonly solidExcessScatterGroup: GPUBindGroup;
  private readonly solidExcessResolveGroup: GPUBindGroup;
  private pipelines?: UniformReferencePipelines;
  private inflowBoundary?: InflowGridBoundary;
  private statsReadback?: GPUBuffer;
  private readbackPending = false;
  private lastTime = 0;
  private referenceVolumeCells = 0;
  /** A ball waiting for a step to wet its cells. See `injectLiquidBall`. */
  private pendingDrop?: InjectedLiquidBall;
  private densityPostProcessing: boolean;
  private densitySharpening: boolean;
  private sharpeningMassCorrection: boolean;
  private gammaDiffusionIterations: number;
  private sharpeningStrength: number;
  private sharpeningDistance: number;
  private solidExcessCorrection: boolean;
  private rigidCoupling: boolean;
  private readonly pressureSchedule: UniformCM11aSchedule;
  /** Host-side cycle budgeting; "fixed" reproduces the pre-P1 command stream. */
  private pressureCycleBudgetLagged: boolean;
  private pressureBudgetHeadroom: number;
  /**
   * Latest asynchronously observed pressure demand. It lags the encoded step
   * by however long the map takes — one to a few frames — which is exactly why
   * the budget rule grows aggressively and shrinks by one cycle at a time.
   */
  private pressureCyclesExecutedSample?: number;
  private pressureCycleConvergedSample = false;
  private pressureCycleDemandReadback?: GPUBuffer;
  private pressureCycleDemandPending = false;
  /**
   * Whether the last observed step ran its recovery finish. Only ever chooses
   * between two launches of the same arithmetic, so a stale answer costs time
   * and nothing else. True until a sample says otherwise.
   */
  private pressureRecoveryExpected = true;
  /**
   * velocityD holds V_face for the geometry as it stands, so the authority
   * pass may rebuild rho' alone. V_face depends on the solid mask, terrain,
   * rigid bodies and scene constants; it is dropped by a scene or settings
   * update, by any step that carries a body, by any store that did not cover
   * the whole lattice, and whenever something else may write velocityD
   * (MacCormack's reverse trace, the stage audit).
   */
  private faceAuthorityStored = false;
  /**
   * The diagnostics reduction is a dense pass whose only reader is readStats:
   * no kernel loads the words it adds. A live frame never maps solver state,
   * so the pass is owed rather than encoded and readStats pays it, once, from
   * the fields the step left behind. A traced step keeps it, because the final
   * phase closes on that pass.
   */
  private diagnosticsReductionOwed = false;
  private stepBodyCount = 0;
  private paperTimeStep: boolean;
  private velocityTransport: GPUVelocityTransport;
  private liquidOnlyVelocityAdvection: boolean;
  private disposed = false;
  private physicsTraceSampleId = 0;
  private physicsTracePending = false;
  private lastPhysicsTraceAt_ms = 0;
  /** One unusable hardware sample retires the stage recorder for this solver;
   * the non-invasive queue-wall observation takes over from then on. */
  private hardwarePhysicsTraceInvalid = false;
  /** UI-selectable A/B control; the environment override keeps Dawn scripts reproducible. */
  private readonly activeRegionEnabled: boolean;
  /** Set by a live scene edit; the next step censuses the whole domain. */
  private activeRegionRescanPending = false;
  /** CPU mirror of the GPU solid bitmask, so a scene edit uploads only the words its pages changed. */
  private readonly solidMask: SolidOccupancyMask;
  /** Timestep whose full-strength inlet support was included in the prior window. */
  private inflowWindowDt = 0;
  /** A/B escape hatch: keep the GPU's indirect records driving every dispatch. */
  private readonly windowDispatchIndirect: boolean;
  private windowReadback?: GPUBuffer;
  private windowReadbackPending = false;
  private windowLagged?: {
    minimum: number[]; maximum: number[]; speed_m_s: number; travel: number[];
  };
  private windowForcedDenseSteps = 0;
  private windowViolations = 0;
  private windowViolationAxes = 0;
  private windowDenseSteps = 0;
  private windowStep = 0;
  private windowReadbackStep = 0;
  private windowLaggedStep = 0;
  private windowMaxLagSteps = 0;
  private windowMainGroups?: [number, number, number];
  private windowVertexGroups?: [number, number, number];

  private constructor(
    private readonly device: GPUDevice,
    public scene: SceneDescription,
    quality: GPUQuality,
    _onRigidLoads?: (loads: GPURigidLoad[]) => void,
    options: WebGPUUniformReferenceOptions = {},
  ) {
    this.geometricVolume = options.geometricVolume === true;
    this.geometricTileWork = this.geometricVolume && options.geometricTileWork !== false;
    this.geometricRedistance = options.geometricRedistance !== false;
    this.volumeDustThreshold = Number.isFinite(options.volumeDustThreshold)
      ? Math.min(1, Math.max(0, options.volumeDustThreshold!)) : 0;
    this.twoLevelVelocity = this.geometricVolume && options.twoLevelVelocity === true;
    this.twoLevelFineReach = Number.isFinite(options.twoLevelFineReach)
      ? Math.round(Math.min(8, Math.max(0, options.twoLevelFineReach!))) : 2;
    this.twoLevelExtensionTiles = options.twoLevelExtensionTiles !== false;
    this.twoLevelAdvectionTiles = options.twoLevelAdvectionTiles !== false;
    this.transportTiles = options.transportTiles !== false;
    // A MARGIN in 4h tiles on top of the reach the step's own measured maximum
    // displacement requires, not a fixed reach: the classify measures that
    // displacement one dispatch before the dilation reads it, so the set
    // tracks the flow instead of being sized for the worst frame of the run.
    // The panel's range starts at zero, the exact predicate. A NEGATIVE margin
    // is a verification-only deficit: it starves the live set below what the
    // measured displacement requires, which is how the restriction is shown to
    // be a restriction at all (the front stalls; nothing is created or lost).
    this.transportReach = Number.isFinite(options.transportReach)
      ? Math.round(Math.min(8, Math.max(-8, options.transportReach!))) : 1;
    this.volumePressureRows = options.volumePressureRows === "all" ? 2 : options.volumePressureRows === true || options.volumePressureRows === "abandoned" ? 1 : 0;
    this.volumeCompaction = options.volumeCompaction === true;
    this.phiSeedFromVolume = options.phiSeedFromVolume === true;
    this.surfaceDeficitBalancing = this.geometricVolume && (options.referenceDimension ?? 3) === 3 && options.surfaceDeficitBalancing !== false;
    this.totalSurfaceVolume = this.geometricVolume && (options.referenceDimension ?? 3) === 3 && options.totalSurfaceVolume !== false;
    this.phiAgreementGain = Number.isFinite(options.phiAgreementGain) ? Math.min(1, Math.max(0, options.phiAgreementGain!)) : 0;
    this.phiAgreementClamp = Number.isFinite(options.phiAgreementClamp) ? Math.min(0.5, Math.max(0, options.phiAgreementClamp!)) : 0.02;
    // Uniform Geometric calls this the SOLVE WINDOW. It needs a positive dust
    // floor for the same reason E3's live set does: the window's diagnostics
    // reduction sums V over the box, which equals the domain sum only while
    // every cell outside the box holds exactly zero, and that is what the
    // floor guarantees. With the floor off the dense schedule is forced.
    this.retainStageDiagnosticsForQA=options.retainStageDiagnosticsForQA===true;
    this.activeRegionEnabled = options.pageDomain !== true && options.activeRegion === true
      && (!this.geometricVolume || this.volumeDustThreshold > 0)
      && (typeof process === "undefined" || process.env.FLUID_UNIFORM_ACTIVE_REGION !== "0");
    // The window's group counts come from the host by default; the GPU's own
    // indirect records remain selectable for the A/B that priced them.
    this.windowDispatchIndirect = typeof process !== "undefined"
      && process.env.FLUID_UNIFORM_WINDOW_DISPATCH === "indirect";
    this.densityPostProcessing = options.densityPostProcessing === true;
    this.densitySharpening = options.densitySharpening !== false;
    this.sharpeningMassCorrection = options.sharpeningMassCorrection !== false;
    this.gammaDiffusionIterations = Math.round(Math.min(UNIFORM_GAMMA_DIFFUSION_MAX_ITERATIONS,
      Math.max(0, options.gammaDiffusionIterations ?? UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS)));
    this.sharpeningStrength = Math.min(this.geometricVolume ? 1 : 2, Math.max(this.geometricVolume ? 0 : 0.25, options.sharpeningStrength ?? 1));
    this.sharpeningDistance = Math.min(3.1, Math.max(0.1, options.sharpeningDistance ?? 2.1));
    this.solidExcessCorrection = options.solidExcessCorrection !== false;
    this.rigidCoupling = options.rigidCoupling !== false;
    this.pressureSchedule = options.pressureSchedule ?? {
      fullCycles: UNIFORM_CM11A_FULL_CYCLES,
      vCycles: UNIFORM_CM11A_V_CYCLES,
      preSweeps: UNIFORM_CM11A_PRE_SWEEPS,
      postSweeps: UNIFORM_CM11A_POST_SWEEPS,
      residualTolerance: 10,
    };
    this.pressureCycleBudgetLagged = options.pressureCycleBudget !== "fixed";
    this.pressureBudgetHeadroom = Number.isFinite(options.pressureBudgetHeadroom)
      ? Math.round(Math.min(4, Math.max(0, options.pressureBudgetHeadroom!)))
      : this.geometricVolume && options.pageDomain ? 0 : UNIFORM_CM11A_DEFAULT_BUDGET_HEADROOM;
    this.paperTimeStep = options.timeStep !== "scene";
    this.velocityTransport = options.velocityTransport === "maccormack"
      ? "maccormack" : "semi-lagrangian";
    this.liquidOnlyVelocityAdvection = options.liquidOnlyVelocityAdvection === true;
    // The stage trace's closing marker pass must dispatch observable work, or
    // Metal skips its end-of-pass timestamp and the first sample retires
    // hardware tracing for this solver. Compile it long before the panel asks.
    void GPUStageTimestampRecorder.prepare(device);
    const [nx, ny, sourceNz] = sceneLatticeDimensions(scene, this.geometricVolume ? Number.MAX_SAFE_INTEGER : device.limits.maxTextureDimension3D);
    const nz = options.referenceDimension === 2 ? 1 : sourceNz;
    if (this.geometricVolume && options.pageDomain) {
      this.pageDomain = initialUniformPageDomain([nx, ny, nz], options.volumePages === 16 ? 16 : 32);
      this.nativePageCoordinates = uniformPageHasNativeCoordinates(this.pageDomain);
      if (!this.nativePageCoordinates) {
        const paged = options.fieldStorageForQA === "paged" || options.phiStorageForQA === "paged" ||
          (options.fieldStorageForQA !== "dense" && !uniformPageHasRectangularCoverage(this.pageDomain));
        // Field-layout QA changes only the atlas layout, keeping its independent storage oracle.
        if (!paged && options.fieldStorageForQA === undefined && options.referenceDimension !== 2 && options.scratchStorageForQA !== "separate")
          this.scratchArena = new UniformScratchArena(device, [nx, ny, nz], nx*ny*nz*UNIFORM_VOLUME_EDGE_BYTES,options.retainStageDiagnosticsForQA);
        this.fieldPages = new UniformTexturePages(device, paged, this.scratchArena);
      }
    }
    this.nativeRootExecution = this.nativePageCoordinates || this.fieldPages?.nativeStorage === true;
    this.volumePageEdge = this.geometricVolume && options.referenceDimension !== 2 && !this.nativePageCoordinates
      ? options.volumePages === "auto" ? (Math.max(nx,ny,nz)>64 ? 32 : 0) : options.volumePages ?? 0
      : 0;
    const paddedPageCells = this.volumePageEdge ? [nx, ny, nz].reduce((n, d) => n * Math.ceil(d / this.volumePageEdge) * this.volumePageEdge, 1) : 0;
    if (options.referenceDimension === 2 && (!this.geometricVolume || scene.container.depthBoundary !== "symmetry" || options.activeRegion))
      throw new Error("2D reference requires geometric mode, symmetry depth and whole-domain work");
    if (this.geometricVolume) {
      if (Math.max(nx,ny,nz)+2 > device.limits.maxTextureDimension3D)
        throw new Error("Uniform Geometric finest lattice exceeds the device texture limit");
      if (!this.volumePageEdge && nx*ny*nz*UNIFORM_VOLUME_EDGE_BYTES > Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize))
        throw new Error("Uniform Geometric finest lattice exceeds the device stencil buffer limit");
      if (nx*ny*nz*24 > Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize))
        throw new Error("Uniform Geometric donor sums exceed the device storage limit");
    }
    // The map lives above the work counters in the third conditioning
    // plane, which is N words. A grid too small to hold both keeps the dense
    // sweeps rather than allocating a plane for a handful of records.
    const tileRecords = Math.ceil(nx / 4) * Math.ceil(ny / 4) * Math.ceil(nz / 4);
    this.sharpenTileCount = this.geometricVolume
      && UNIFORM_VOLUME_SHARPEN_TILE_MAP_WORD + tileRecords <= nx * ny * nz ? tileRecords : 0;
    // The 4h face table and class map sit in the second conditioning plane,
    // whose per-step clears are ranged away from it. Same size test, plus the
    // shell counter above the two ping-pong planes. Every axis must be a
    // multiple of four: the sampler's convention is "fine face 4t+3 is coarse
    // face t", which is what the extension hierarchy's own transfer computes
    // only when the coarse level tiles the lattice exactly.
    this.twoLevelTileCount = this.geometricVolume
      && (options.referenceDimension === 2 ? [nx, ny] : [nx, ny, nz]).every((value) => value % 4 === 0)
      && UNIFORM_VOLUME_TWO_LEVEL_WORDS_PER_TILE * tileRecords
        + UNIFORM_VOLUME_TWO_LEVEL_COUNTER_WORDS <= nx * ny * nz ? tileRecords : 0;
    // One tile covers the finest trilinear tap, which reaches one cell below a
    // fine tile. The FIM's accurate band is two of the LARGEST cells, so on an
    // anisotropic lattice it spans 2*max(h)/min(h) of the smallest; the shell
    // must contain that too, measured from a seed tile's own boundary.
    const spacing = [scene.container.width_m / nx, scene.container.height_m / ny,
      scene.container.depth_m / nz];
    this.twoLevelShellReach = Number.isFinite(options.twoLevelShellReach)
      ? Math.round(Math.min(8, Math.max(0, options.twoLevelShellReach!)))
      : Math.max(1, Math.ceil(0.5 * Math.max(...spacing) / Math.min(...spacing)));
    const allocation = planUniformHostAllocation(nx, ny, nz, "maccormack");
    this.negativeBoundaryVelocityBytes = allocation.boundaryVelocityBytes;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const texture3d = (label: string, format: GPUTextureFormat, size: GPUExtent3D, native=false) =>
      this.fieldPages ? this.fieldPages.createTexture({ label, size, dimension: "3d", format, usage },native)
        : device.createTexture({ label, size, dimension: "3d", format, usage });
    const velocity = (label: string) => texture3d(label, "rgba32float", allocation.velocityExtent);
    const scalar = (label: string) => texture3d(label, "r32float", allocation.volumeExtent);
    this.velocityA = velocity("Uniform reference velocity A");
    this.velocityB = velocity("Uniform reference velocity B");
    const lazyMac = !!this.scratchArena && this.velocityTransport !== "maccormack"
      && !(typeof process !== "undefined" && process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT === "1");
    this.velocityC = lazyMac ? texture3d("Uniform reference unused velocity C", "rgba32float", [1,1,1]) : velocity("Uniform reference velocity C");
    this.velocityD = velocity("Uniform reference velocity D");
    const boundaryVelocity = (label: string) => device.createBuffer({ label, size: allocation.boundaryVelocityBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.boundaryVelocityA = boundaryVelocity("Uniform reference negative boundary velocity A");
    this.boundaryVelocityB = boundaryVelocity("Uniform reference negative boundary velocity B");
    this.boundaryVelocityC = boundaryVelocity("Uniform reference negative boundary velocity C");
    this.boundaryVelocityD = boundaryVelocity("Uniform reference negative boundary velocity D");
    if (typeof process !== "undefined" && process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT === "1") {
      this.symmetryStageAuditNegativeBoundaryVelocity = device.createBuffer({
        label: "Uniform audit pre-extrapolation negative boundary velocity",
        size: allocation.boundaryVelocityBytes,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }
    // Geometric uses CM11a's own pressure hierarchy. Legacy density diffusion
    // is disabled, so the compatibility pressure slots need only valid views.
    const pressure=(label:string)=>this.geometricVolume ? texture3d(label,"r32float",[1,1,1]) : scalar(label);
    this.pressureA = pressure("Uniform reference pressure A");
    this.pressureB = pressure("Uniform reference pressure B");
    this.volumeA = scalar("Uniform reference volume A");
    this.volumeB = scalar("Uniform reference volume B");
    this.surfaceA = scalar("Uniform reference smoothed surface A");
    this.surfaceB = scalar("Uniform reference smoothed surface B");
    this.gammaA = scalar("Uniform reference transport gamma A");
    this.gammaB = scalar("Uniform reference transport gamma B");
    if (this.geometricVolume) {
      this.vertexPhiField = texture3d("Uniform Geometric vertex phi", "r32float", [nx + 1, ny + 1, nz + 1],options.phiStorageForQA !== "paged");
      this.vertexPhiScratch = texture3d("Uniform Geometric vertex phi scratch", "r32float", [nx + 1, ny + 1, nz + 1],options.phiStorageForQA !== "paged");
      const edgeBytes = this.volumePageEdge && !this.nativeRootExecution ? paddedPageCells * UNIFORM_VOLUME_EDGE_BYTES : nx * ny * nz * UNIFORM_VOLUME_EDGE_BYTES;
      if (edgeBytes > device.limits.maxStorageBufferBindingSize || edgeBytes > device.limits.maxBufferSize)
        throw new Error(`Uniform Geometric receiver stencils require ${edgeBytes} bytes, exceeding the device limit`);
      this.denseLevelSetVolumeSource = { vertexPhi: this.present(this.vertexPhiField), openFraction: this.present(this.gammaB),
        cellSize_m: [scene.container.width_m/nx, scene.container.height_m/ny, scene.container.depth_m/nz] };
      // Smaller than the edge buffer checked above: six 32-bit limbs/cell.
      this.volumeDonorSums = this.scratchArena?.buffer ?? device.createBuffer({ label: "Uniform Geometric exact donor sums", size: nx*ny*nz*24,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.volumeEdges = this.scratchArena?.buffer ?? device.createBuffer({ label: "Uniform Geometric nine-donor stencils", size: edgeBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    }
    this.heightA = device.createTexture({ label: "Uniform reference column base", size: [nx, nz], format: "rg32float", usage });
    this.heightB = device.createTexture({ label: "Uniform reference column occupancy", size: [nx, nz], format: "rg32float", usage });
    this.terrainTexture = device.createTexture({ label: "Uniform reference terrain", size: [nx, nz], format: "r32float", usage });
    const transportUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    this.transportA = (this.fieldPages ?? device).createTexture({ label: "Uniform reference transport A", size: allocation.transportExtent, dimension: "3d", format: "rgba32float", usage: transportUsage });
    this.transportB = (this.fieldPages ?? device).createTexture({ label: "Uniform reference transport B", size: lazyMac ? [1,1,1] : allocation.transportExtent, dimension: "3d", format: "rgba32float", usage: transportUsage });
    if (typeof process !== "undefined" && process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT === "1") {
      this.symmetryStageAuditFields = Object.freeze({
        preExtrapolationVelocity: velocity("Uniform audit velocity before current Sec. 3.3"),
        previousRawDensity: scalar("Uniform audit raw density before current Sec. 3.4"),
        extrapolationDensityAuthority: scalar("Uniform audit rho-prime for current Sec. 3.3"),
        densityAdvection: scalar("Uniform audit density after advection"),
        densityDiffusion: scalar("Uniform audit density after gamma diffusion"),
        densitySharpening: scalar("Uniform audit density after sharpening"),
        gammaPostAdvection: scalar("Uniform audit gamma after Sec. 3.4 advection"),
        gammaPostDiffusion: scalar("Uniform audit gamma after diffusion"),
        velocityPrediction: velocity("Uniform audit velocity after forward prediction"),
        predictedExtrapolation: this.transportB,
        reverseAdvection: this.velocityD,
        velocityAdvection: velocity("Uniform audit velocity after configured advection"),
        pressureProjection: velocity("Uniform audit velocity after pressure projection"),
      });
    }
    this.params = device.createBuffer({ label: "Uniform reference parameters", size: 208, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.solidMask = new SolidOccupancyMask([nx, ny, nz]);
    this.solidMask.update(solidWorldForScene(scene));
    const packedSolidVoxels = this.solidMask.words;
    const activeRegionBytes = UNIFORM_ACTIVE_HEADER_WORDS * 4;
    this.activeRegion = device.createBuffer({
      label: "Uniform reference active liquid region", size: activeRegionBytes+(this.pageDomain?.words.byteLength??0),
      // readStats copies the published bounds/counters into its readback
      // packet in both dense and sparse modes.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const activeSummaryCount = Math.ceil(nx / 4) * Math.ceil(ny / 4) * Math.ceil(nz / 4);
    const phiWindow = this.fieldPages && this.pageDomain && uniformPageHasRectangularCoverage(this.pageDomain)
      && options.phiStorageForQA !== "paged" && options.phiWindowForQA !== false && options.phiReadAuditForQA !== true && this.volumeDustThreshold > 0;
    if(phiWindow){
      this.phiRegion=device.createBuffer({label:"Compiled phi execution region",size:this.activeRegion.size,
        usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      this.phiDispatch=device.createBuffer({label:"Compiled phi vertex dispatch",size:activeRegionBytes,
        usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(this.phiRegion,activeRegionBytes,this.pageDomain!.words.buffer);
    }
    const activeSummaryBytes = this.pageDomain && !phiWindow ? 0 : activeSummaryCount * UNIFORM_ACTIVE_SUMMARY_BYTES;
    this.solidVoxelScratchOffsetWords = (activeRegionBytes + activeSummaryBytes) / 4;
    this.activeScratch = device.createBuffer({
      label: "Uniform reference active liquid census scratch and summaries",
      size: activeRegionBytes + (this.volumeTransportPageView?.size ?? 0) + activeSummaryBytes + packedSolidVoxels.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.activeScratch, this.solidVoxelScratchOffsetWords * 4,
      packedSolidVoxels.buffer as ArrayBuffer, packedSolidVoxels.byteOffset,
      packedSolidVoxels.byteLength);
    this.activeDispatch = device.createBuffer({
      label: "Uniform reference active indirect dispatches", size: activeRegionBytes,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    if(this.pageDomain){
      device.queue.writeBuffer(this.activeRegion,activeRegionBytes,this.pageDomain.words.buffer);
      this.pageDomainPublication=new UniformPageDomainPublication(device,this.pageDomain,this.activeRegion);
      if (!this.nativeRootExecution) this.pageDomainDispatch=this.pageDomainPublication.dispatch;
      this.pageDomainView=this.pageDomainPublication.view;
    }
    // Created before the extrapolator: the extension binds it read_write to
    // read the tile classes and to publish the 4h face table the sampler reads.
    const balanceRecords = this.pageDomain && !this.nativeRootExecution ? this.pageDomain.capacity * (this.pageDomain.edge / 4) ** 3 : tileRecords;
    this.surfaceDeficitBalanceBytes = this.geometricVolume ? (2 + 2 * balanceRecords) * 4 : 0;
    const pageBaseBytes = (this.scratchArena?.conditioningBytes ?? allocation.conditioningBytes) + this.surfaceDeficitBalanceBytes;
    if (this.volumePageEdge) this.volumePageConfig = {
      edge: this.volumePageEdge, base: pageBaseBytes / 4,
      nativeRecords: this.nativeRootExecution,
      work: Math.max(nx,ny,nz)>64 && options.volumePageWork !== false,
      count: [nx, ny, nz].reduce((n, d) => n * Math.ceil(d / this.volumePageEdge), 1),
    };
    const pageBytes = this.volumePageConfig ? 4 * (8 + 2 * this.volumePageConfig.count + (this.volumePageConfig.work ? 1 + tileRecords : 0)) : 0;
    if(this.volumePageConfig){
      this.volumeTransportPageView=device.createBuffer({label:"Transport page activity",size:4*(8+2*this.volumePageConfig.count),usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      this.volumePageSharpenFlag=device.createBuffer({label:"Uniform sharpening phase flag",size:4,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(this.volumePageSharpenFlag,0,new Uint32Array([1]));
    }
    if(this.volumePageConfig?.work){
      this.volumeWorkDispatch=device.createBuffer({label:"Uniform volume tile dispatch",size:12,usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(this.volumeWorkDispatch,0,new Uint32Array([0,1,1]));
      this.volumeWorkCounts=device.createBuffer({label:"Uniform volume work counters",size:8,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    }
    const fullLattice = !this.activeRegionEnabled && uniformAbOn("staticid");
    const source = this.geometricVolume ? createUniformReferenceComputeShader(true, options.referenceDimension ?? 3, this.volumePageConfig,this.nativeRootExecution ? undefined : this.pageDomain, fullLattice) : uniformReferenceComputeShader;
    const fixedFields = new Map([
      [0,this.velocityA],[1,this.velocityB],[3,this.pressureB],[4,this.volumeA],[5,this.volumeB],
      [12,this.velocityC],[13,this.velocityD],[14,this.transportA],[16,this.surfaceA],
      [20,this.surfaceA],[24,this.gammaA],[25,this.gammaB],[31,this.vertexPhiField!],[32,this.vertexPhiScratch!],
    ]);
    // These bindings always hold retained native textures, even when ping-pong
    // groups change. Keep their interpolation free of scratch address branches.
    const nativeBindings = new Set([0,1,2,3,4,5,12,13,14,16,20,21,24,25,31,32]);
    this.shaderSource = this.fieldPages?.shader(source,fixedFields,!!this.pageDomain && !this.nativeRootExecution,this.nativeRootExecution,false,nativeBindings) ?? source;
    if (this.fieldPages && options.phiStorageForQA !== "paged" && this.pageDomain && uniformPageHasRectangularCoverage(this.pageDomain)) {
      // The accepted catalogue is currently the complete rectangular domain.
      // Phi owns native persistent fields; no per-step atlas packing is needed.
      const phiSource = options.phiReadAuditForQA === true
        ? createUniformReferenceComputeShader(true,options.referenceDimension ?? 3,this.volumePageConfig,this.pageDomain,fullLattice&&!this.phiRegion)
          .replace("return pageDomainVertex(gid);", "return vec3i(gid);")
        : this.phiRegion
        // phiRegion replaces binding 29 with a census-driven window of its own,
        // so these kernels keep reading the header whatever the solver's mode.
        ? createUniformReferenceComputeShader(true,options.referenceDimension ?? 3,this.volumePageConfig)
        : source.replace("return pageDomainVertex(gid);", "return vec3i(gid);");
      // Native literal interpolation keeps dense throughput. Runtime loops did
      // not reduce the frozen-input rounding discrepancy at the high dust floor.
      this.phiShaderSource = this.fieldPages.shader(phiSource,fixedFields,options.phiReadAuditForQA === true,options.phiLiteralLoopsForQA !== false,true,nativeBindings);
      // Preserve the iterative closest-point loop form: full unrolling changes
      // near-degenerate injected-surface cases under Metal optimization.
      this.phiRedistanceShaderSource = this.fieldPages.shader(phiSource,fixedFields,options.phiReadAuditForQA === true,false,true,nativeBindings);
    }
    this.conditioningScratch = device.createBuffer({ label: "Uniform reference compatibility scratch", size: pageBaseBytes + pageBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.velocityExtrapolator = new WebGPUUniformVelocityExtrapolator(
      device, [nx, ny, nz], [
        scene.container.width_m / nx,
        scene.container.height_m / ny,
        scene.container.depth_m / nz,
      ], this.params, this.surfaceA, this.velocityD,
      this.velocityA, this.velocityC, this.transportA, this.transportB,
      this.activeRegion, this.conditioningScratch,
      this.activeRegionEnabled ? this.activeDispatch : undefined,
      options.sourceAwareExtension ?? this.geometricVolume, options.fuseExtensionPack, this.fieldPages, this.nativeRootExecution ? undefined : this.pageDomain, this.pageDomainDispatch,
    );
    // Without a ceil(n/4) hierarchy level there is no 4h field to sample.
    if (!this.velocityExtrapolator.coarseVelocityTableAvailable) this.twoLevelTileCount = 0;
    if (this.twoLevelTileCount === 0) this.twoLevelVelocity = false;
    // The E1 table: four words per 4h tile from word N -- three 4h faces, then
    // the class word the fine-tiles view reads. N*4 bytes is a multiple of 256,
    // the storage offset alignment, because every axis is a multiple of four.
    if (this.twoLevelTileCount > 0) this.tileClassRecords = { records: {
      buffer: this.conditioningScratch, offset: this.scratchArena ? 0 : 4 * nx * ny * nz, size: 16 * this.twoLevelTileCount,
    } };
    this.extrapolationActiveFrontPassCeiling = this.velocityExtrapolator.activeFrontPassCeiling;
    if (options.extensionFrontSweeps !== undefined) this.velocityExtrapolator.setFrontPasses(options.extensionFrontSweeps);
    this.reductions = device.createBuffer({ label: "Uniform reference diagnostics and volume control", size: 40, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (typeof process !== "undefined" && process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT === "1") {
      this.symmetryStageAuditBetaBuffer = device.createBuffer({
        label: "Uniform audit Sec. 3.4 beta",
        size: nx * ny * nz * 4,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }
    if (typeof process !== "undefined"
      && process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT === "1") {
      this.symmetryStageAuditMacCormackBuffer = device.createBuffer({
        label: "Uniform audit bounded MacCormack decisions",
        size: nx * ny * nz * 3 * 8 * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
    }
    this.macCormackAuditBinding = this.symmetryStageAuditMacCormackBuffer ?? device.createBuffer({
      label: "Uniform disabled MacCormack audit binding",
      // One vec4 is the minimum valid runtime-array storage resource. It is
      // never accessed in this shader variant and must not alias binding 19.
      size: 16,
      usage: GPUBufferUsage.STORAGE,
    });
    this.rigidExchange = device.createBuffer({ label: "Uniform reference rigid exchange", size: GPU_RIGID_EXCHANGE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.rigidSystem = new WebGPURigidBodySystem(device, scene, this.rigidExchange,
      this.terrainTexture);
    this.rigidSystem.syncBodies(initializeRigidBodies(scene.rigidBodies));
    this.inflowBoundary = scene.fluid.inflow
      ? createInflowGridBoundary(scene.fluid.inflow, scene.container, [nx, ny, nz]) : undefined;

    const geometricLayout: GPUBindGroupLayoutEntry[] = this.geometricVolume ? [
      { binding: 31, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 32, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      ...(!this.scratchArena ? [{ binding: 33, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } }] : []),
    ] : [];
    const mainEntries: GPUBindGroupLayoutEntry[] = [
      ...geometricLayout,
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "2d" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 14, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 15, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 19, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 20, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 21, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 24, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 25, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 26, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 27, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 28, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 29, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 30, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ];
    if(this.fieldPages) mainEntries.push(...this.fieldPages.layout([]));
    this.mainLayout = device.createBindGroupLayout({ entries: mainEntries });
    this.pressureInputLayout = this.geometricVolume ? device.createBindGroupLayout({
      entries: mainEntries.filter(e => e.binding !== 32 && e.binding !== 33 && (!this.scratchArena || e.binding !== 28)),
    }) : this.mainLayout;
    // Residency owns the domain, not the packing of multigrid scratch. Native
    // fields retain the numerical operators without translating every tap.
    const pagedPressure = options.pressureStorageForQA !== undefined;
    this.pressureMultigrid = new WebGPUUniformPressureMultigrid(device, [nx, ny, nz], [
      scene.container.width_m / nx,
      scene.container.height_m / ny,
      scene.container.depth_m / nz,
    ], this.pressureSchedule, this.activeRegionEnabled ? this.activeDispatch : undefined,
      undefined, false, options.referenceDimension ?? 3, options.pressureCycleDispatch === "indirect" ||
        (options.pressureCycleDispatch !== "direct" && pagedPressure),
      pagedPressure, options.pressureStorageForQA === "paged-logical",
      uniformAbOn("inplace") && (options.referenceDimension ?? 3) === 3
        && scene.container.depthBoundary !== "symmetry", this.scratchArena && !pagedPressure ? this.fieldPages : undefined, this.geometricVolume && options.pressureSmoothingForQA !== "dense");
    this.pressureWindowCapacity = [nx, ny, nz];
    this.pressureDomainKey = this.pressureWindowCapacity.join("x");
    this.pressureInstances.set(this.pressureDomainKey, this.pressureMultigrid);
    this.pressureWindowLattice = this.geometricVolume && this.activeRegionEnabled
      && !this.windowDispatchIndirect && options.pressureWindow !== false;
    this.mainPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.mainLayout] });
    const sampler = device.createSampler({ minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    const view = (texture: GPUTexture) => this.fieldPages?.view(texture) ?? texture.createView();
    const group = (velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture,
      pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture,
      heightIn: GPUTexture, heightOut: GPUTexture, predicted = velocityIn,
      reversed = velocityIn, transport = this.transportA, surface = volumeIn,
      gammaRead = this.gammaA, gammaWrite = this.gammaB,
      boundaryRead = this.boundaryVelocityA, boundaryWrite = this.boundaryVelocityB,
      velocityPhase = volumeIn, reversePhi = false, pressureOnly = false, donorSums = false) => this.createPageAwareGroup({
        layout: pressureOnly ? this.pressureInputLayout : this.mainLayout, entries: ([
          ...(this.geometricVolume ? [
            { binding: 31, resource: view((reversePhi ? this.vertexPhiScratch! : this.vertexPhiField!)) },
            { binding: 32, resource: view((reversePhi ? this.vertexPhiField! : this.vertexPhiScratch!)) },
            ...(!this.scratchArena ? [{ binding: 33, resource: { buffer: this.volumeEdges! } }] : []),
          ] : []),
          { binding: 0, resource: view(velocityIn) }, { binding: 1, resource: view(velocityOut) },
          { binding: 2, resource: view(pressureIn) }, { binding: 3, resource: view(pressureOut) },
          { binding: 4, resource: view(volumeIn) }, { binding: 5, resource: view(volumeOut) },
          { binding: 6, resource: { buffer: this.params } }, { binding: 7, resource: view(heightIn) },
          { binding: 8, resource: view(heightOut) }, { binding: 9, resource: { buffer: this.reductions } },
          { binding: 10, resource: { buffer: this.rigidSystem.stateBuffer } }, { binding: 11, resource: { buffer: donorSums ? this.volumeDonorSums! : this.rigidExchange, ...(donorSums && this.scratchArena ? {offset:this.scratchArena.donorOffset,size:this.scratchArena.donorBytes}: {}) } },
          { binding: 12, resource: view(predicted) }, { binding: 13, resource: view(reversed) },
          { binding: 14, resource: view(transport) }, { binding: 15, resource: sampler },
          { binding: 16, resource: view(velocityPhase) },
          { binding: 19, resource: { buffer: this.conditioningScratch } },
          { binding: 20, resource: view(surface) }, { binding: 21, resource: view(this.terrainTexture) },
          { binding: 24, resource: view(gammaRead) }, { binding: 25, resource: view(gammaWrite) },
          { binding: 26, resource: { buffer: boundaryRead } }, { binding: 27, resource: { buffer: boundaryWrite } },
          { binding: 28, resource: { buffer: this.macCormackAuditBinding } },
          { binding: 29, resource: { buffer: this.activeRegion } },
          { binding: 30, resource: { buffer: this.activeScratch } },
        ] as GPUBindGroupEntry[]).filter(e => !pressureOnly || (e.binding !== 32 && e.binding !== 33 && (!this.scratchArena || e.binding !== 28))),
      },donorSums?this.scratchArena?.edgeBytes:undefined);
    // Pressure setup and projection never read the reverse-advection slot, so
    // under Geometric it carries this advance's V_face authority instead.
    const sharedFaceOpen = this.geometricVolume ? this.velocityD : this.velocityB;
    this.extrapolationAuthorityGroup = group(
      this.velocityA, this.velocityD, this.pressureA, this.pressureB,
      this.volumeA, this.surfaceA, this.heightB, this.heightA,
    );
    this.semiLagrangianGroup = group(
      this.velocityA, this.velocityB, this.pressureA, this.pressureB,
      this.volumeB, this.volumeA, this.heightB, this.heightA,
      this.velocityA, this.velocityA, this.transportA, this.volumeB,
      this.gammaA, this.gammaB, this.boundaryVelocityA, this.boundaryVelocityB,
      this.surfaceA,
    );
    this.advectGroup = group(this.velocityA, this.velocityC, this.pressureA, this.pressureB,
      this.volumeA, this.volumeB, this.heightB, this.heightA,
      this.velocityB, this.velocityD, this.transportA, this.volumeA,
      this.gammaA, this.gammaB, this.boundaryVelocityA, this.boundaryVelocityC,
      this.surfaceA);
    this.reverseGroup = group(this.velocityC, this.velocityD, this.pressureA, this.pressureB,
      this.volumeA, this.volumeB, this.heightB, this.heightA,
      this.velocityA, this.velocityB, this.transportB, this.volumeA,
      this.gammaA, this.gammaB, this.boundaryVelocityC, this.boundaryVelocityD,
      this.surfaceA);
    this.correctGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB,
      this.volumeB, this.volumeA, this.heightB, this.heightA,
      this.velocityC, this.velocityD, this.transportA, this.volumeB,
      this.gammaA, this.gammaB, this.boundaryVelocityA, this.boundaryVelocityB,
      this.surfaceA);
    this.pressureMultigridGroup = group(this.velocityB, this.velocityA, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA, this.velocityB, sharedFaceOpen, this.transportA, this.volumeB, this.gammaA, this.gammaB, this.boundaryVelocityB, this.boundaryVelocityA, this.volumeB, false, this.geometricVolume);
    // Rebuilt whenever the active CM11a instance changes: this is the only
    // bind group that names the hierarchy's finest pressure texture.
    this.makeProjectGroup = (pressure: GPUTexture) => group(this.velocityB, this.velocityA, pressure, this.pressureA, this.volumeB, this.volumeA, this.heightB, this.heightA, this.velocityB, sharedFaceOpen, this.transportA, this.volumeB, this.gammaA, this.gammaB, this.boundaryVelocityB, this.boundaryVelocityA);
    this.projectGroup = this.makeProjectGroup(this.pressureMultigrid.pressureTexture);
    this.rigidGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityA, this.velocityA, this.transportA, this.volumeA, this.gammaA, this.gammaB, this.boundaryVelocityA, this.boundaryVelocityB);
    this.reductionGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.densityTraceGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA,
      this.velocityA, this.velocityA, this.transportA, this.volumeA, this.gammaA, this.gammaB);
    // Paper Sec. 3.4 step 7 scatters the deficit from pre-advection gamma^n.
    this.densityScatterGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA,
      this.velocityA, this.velocityA, this.transportA, this.volumeA, this.gammaA, this.gammaB);
    // Donor passes do not exchange rigid impulses. Reuse that binding slot
    // to stay within the adapter's ten-storage-buffer stage limit.
    this.volumeDonorGroup = this.geometricVolume ? group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA,
      this.velocityA, this.velocityA, this.transportA, this.volumeA, this.gammaA, this.gammaB, this.boundaryVelocityA, this.boundaryVelocityB, this.volumeA, false, false, true) : this.densityTraceGroup;
    this.densityGatherGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA,
      this.velocityA, this.velocityA, this.transportA, this.volumeA, this.gammaB, this.gammaA);
    this.gammaDiffusionGroups = [
      group(this.velocityA, this.velocityB, this.pressureA, this.pressureB,
        this.volumeB, this.volumeA, this.heightB, this.heightA,
        this.velocityA, this.velocityA, this.transportA, this.volumeB,
        this.gammaA, this.gammaB),
      group(this.velocityA, this.velocityB, this.pressureA, this.pressureB,
        this.volumeA, this.volumeB, this.heightB, this.heightA,
        this.velocityA, this.velocityA, this.transportA, this.volumeA,
        this.gammaB, this.gammaA),
    ];
    this.sharpenComputeGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.sharpenScatterGroup = group(this.velocityA, this.velocityB, this.pressureB, this.pressureA, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.sharpenResolveGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    // A moving body can cover an occupied cell between advances. Reconcile
    // the current density against the newly uploaded body geometry before the
    // conservative operator masks solid donors. The later pair retains the
    // paper's post-sharpening rho <= V cleanup.
    this.solidEntryScatterGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.solidEntryResolveGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.solidExcessScatterGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.solidExcessResolveGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.postprocessBlurXGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.surfaceA, this.heightB, this.heightA);
    this.postprocessBlurYGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.surfaceA, this.surfaceB, this.heightB, this.heightA);
    this.postprocessBlurZGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.surfaceB, this.surfaceA, this.heightB, this.heightA);
    this.postprocessResolveGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.surfaceB, this.heightB, this.heightA,
      this.velocityA, this.velocityA, this.transportA, this.surfaceA);
    this.wallFilmResolveGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.surfaceB, this.heightB, this.heightA);
    if (this.geometricVolume) this.phiReverseGroup = group(
      this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB,
      this.heightB, this.heightA, this.velocityA, this.velocityA, this.transportA, this.volumeA,
      this.gammaA, this.gammaB, this.boundaryVelocityA, this.boundaryVelocityB, this.volumeA, true);
    if(this.phiRegion)for(const original of [this.densityTraceGroup,this.densityGatherGroup,this.phiReverseGroup!,this.reductionGroup]){
      const descriptor=this.groupDescriptors.get(original)!;
      this.phiGroups.set(original,this.createPageAwareGroup({...descriptor,entries:[...descriptor.entries].map(entry=>
        entry.binding===29?{binding:29,resource:{buffer:this.phiRegion!}}:entry)}));
    }

    const count = nx * ny * nz;
    // The legacy host plan reserves nine scalars; this owner has eight. Add
    // the extension hierarchy/origins and the two RG columns plus terrain,
    // which that plan does not include. Scratch backing is reconciled below.
    const hostAuxiliaryBytes = this.velocityExtrapolator.scratchBytes
      - (nx+2)*(ny+2)*(nz+2)*6*16 - count*4 + nx*nz*20;
    this.info = {
      nx, ny, nz, storedNy: ny, cellCount: count, equivalentUniformCells: count,
      compressionRatio: 1, activeCompressionRatio: 1, activeSampleCount: count,
      regularLayers: ny, maximumNeighborDelta: 0, gridKind: "uniform",
      cellSize_m: Math.min(scene.container.width_m / nx, scene.container.height_m / ny, scene.container.depth_m / nz),
      pressureIterations: 0, pressureSolver: `CM11a ${pagedPressure ? "paged (QA)" : this.pageDomain ? "native-page" : "dense"} LCP multigrid (${this.pressureSchedule.fullCycles} Full-Cycles + ${this.pressureSchedule.vCycles} V-Cycles, ${this.pressureSchedule.preSweeps}/${this.pressureSchedule.postSweeps} pre/post PRBGS)`,
      allocatedBytes: allocation.allocatedBytes - (this.geometricVolume ? count*8-8 : 0) - (this.scratchArena ? allocation.conditioningBytes-this.scratchArena.conditioningBytes : 0) - (lazyMac ? nx*ny*nz*16+(nx+2)*(ny+2)*(nz+2)*16-32 : 0) + 8 + pageBytes + (this.volumeWorkDispatch?20:0) + (this.volumePageSharpenFlag?4:0) + this.surfaceDeficitBalanceBytes + this.pressureMultigrid.allocatedBytes
        + (this.geometricVolume ? 8 * (nx+1)*(ny+1)*(nz+1) + (this.scratchArena ? 0 : count*24 + (this.volumeEdges?.size ?? 0)) + 12 : 0)
        + activeRegionBytes * 3 + (this.phiRegion?.size ?? 0) + (this.phiDispatch?.size ?? 0) + (this.pageDomain ? this.pageDomain.words.byteLength + 32 + (this.pageDomainView?.size??0) : 0) + (this.volumeTransportPageView?.size ?? 0) + activeSummaryBytes + packedSolidVoxels.byteLength
        + hostAuxiliaryBytes + (this.symmetryStageAuditMacCormackBuffer ? 0 : 16), quality,
      submittedTime_s: 0, simulatedTime_s: 0, completedTime_s: 0,
      simulationLag_s: 0, encodedSteps: 0, maximumTallCellHeight: 0,
      volumeControl: true,
      hostFluidAuthority: "gpu-resident", hostSimulationSizedWorkItems: 0,
      hostSchedulingUsesReadback: this.pressureCycleBudgetLagged && this.pressureMultigrid.residualTolerance > 0,
      ...(this.pageDomain?{uniformDomainAuthority:"pages" as const,uniformDomainPages:this.pageDomain.count,
        ...(this.nativePageCoordinates ? {uniformVolumePageEdge:this.pageDomain.edge,uniformVolumePagesTotal:1,uniformVolumePagesActive:1} : {}),
        uniformDomainMigration:this.nativePageCoordinates ? "Native single-page coordinates; all-resident domain" : pagedPressure ? "Paged fluid and pressure fields (QA); all-resident domain" : this.phiShaderSource ? "Native rectangular fields; all-resident page catalogue" : "Paged fluid fields; native pressure workspace; all-resident domain"}:{}),
      ...(this.volumePageConfig ? { uniformVolumePageEdge: this.volumePageEdge, uniformVolumePagesTotal: this.volumePageConfig.count, uniformVolumePageBytes: this.scratchArena?.edgeBytes ?? this.volumeEdges!.size } : {}),
    };
    this.volumeTexture = this.present(this.volumeA);
    this.columnBaseTexture = this.heightA;
    this.velocityTexture = this.present(this.velocityA);
    this.extrapolatedVelocityTexture = this.present(this.transportA);
    if(!this.scratchArena){
      this.present(this.velocityB);
      this.present(this.velocityExtrapolator.activeStateTexture);
    }
    if (this.geometricVolume && (options.referenceDimension ?? 3) === 3) {
      this.surfaceVolumeCorrection = new UniformSurfaceVolumeCorrection(device, [nx,ny,nz],
        [scene.container.width_m/nx,scene.container.height_m/ny,scene.container.depth_m/nz],
        this.vertexPhiField!,this.volumeB,this.gammaB,this.fieldPages);
    }
    if(this.fieldPages) { this.present(this.surfaceB); if(!this.scratchArena)this.present(this.vertexPhiScratch!); this.present(this.gammaA); }
    if(this.symmetryStageAuditFields)this.symmetryStageAuditTextures=Object.freeze(Object.fromEntries(
      Object.entries(this.symmetryStageAuditFields).map(([name,field])=>[name,this.present(field)]),
    )) as NonNullable<WebGPUUniformReferenceSolver["symmetryStageAuditTextures"]>;
    this.initializeVolumeAndTerrain();
  }

  static async createAsync(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    options: WebGPUUniformReferenceOptions,
    onProgress: GPUInitializationReporter,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<WebGPUUniformReferenceSolver> {
    const runner = new GPUInitializationTaskRunner(onProgress, signal);
    let solver: WebGPUUniformReferenceSolver | undefined;
    try {
      await runner.run([{ id: "uniform.allocate", phase: "allocation", label: "Allocate uniform reference resources", run: () => {
        solver = new WebGPUUniformReferenceSolver(device, scene, quality, onRigidLoads, { ...options, deferPipelineCompilation: true });
      } }]);
      await runner.run(solver!.initializationTasks(signal));
      return solver!;
    } catch (error) {
      solver?.destroy();
      throw error;
    }
  }

  private initializationTasks(signal?: AbortSignal): GPUInitializationTask[] {
    const tasks = [...this.rigidSystem.initializationTasks()];
    const compiler = gpuCompilationManagerFor(this.device);
    const shaderModule = compiler.createShaderModule({ label: "Uniform reference kernels", code: this.shaderSource });
    const phiModule = this.phiShaderSource ? compiler.createShaderModule({ label: "Native vertex phi kernels", code: this.phiShaderSource }) : shaderModule;
    const phiRedistanceModule = this.phiRedistanceShaderSource ? compiler.createShaderModule({label:"Native phi redistance kernels",code:this.phiRedistanceShaderSource}) : shaderModule;
    const compiled: Partial<UniformReferencePipelines> = {};
    const ids = PIPELINES.map(([key]) => `uniform.pipeline.${key}`);
    PIPELINES.forEach(([key, label, entryPoint], index) => tasks.push({
      id: ids[index], phase: "solver-pipelines", label,
      run: async () => {
        compiled[key] = await compiler.compileComputePipeline({
          label: `Uniform reference - ${entryPoint}`,
          layout: this.mainPipelineLayout,
          compute: { module: shaderModule, entryPoint },
        }, { priority: "visible", signal });
      },
    }));
    if (this.geometricVolume) for (const entryPoint of UNIFORM_VOLUME_ENTRIES) {
      const id = `uniform.volume.${entryPoint}`; ids.push(id);
      tasks.push({ id, phase: "solver-pipelines", label: entryPoint, run: async () => {
        this.volumePipelines[entryPoint] = await compiler.compileComputePipeline({
          label: `Uniform Geometric - ${entryPoint}`, layout: this.mainPipelineLayout,
          compute: { module: entryPoint === "uvAdvectPhi" ? phiModule : entryPoint === "uvRedistancePhi" ? phiRedistanceModule : shaderModule, entryPoint },
        }, { priority: "visible", signal });
      } });
    }
    if (this.volumePageConfig) for (const entryPoint of UNIFORM_VOLUME_PAGE_ENTRIES) {
      const id = `uniform.pages.${entryPoint}`; ids.push(id);
      tasks.push({ id, phase: "solver-pipelines", label: entryPoint, run: async () => {
        this.pagePipelines[entryPoint] = await compiler.compileComputePipeline({
          label: `Uniform pages - ${entryPoint}`, layout: this.mainPipelineLayout,
          compute: { module: shaderModule, entryPoint },
        }, { priority: "visible", signal });
      } });
    }
    // One module, two specializations of the same four sweeps: the work map is
    // a pipeline-overridable constant, so the live toggle only picks a pipeline.
    if (this.sharpenTileCount > 0) {
      for (const entryPoint of [...UNIFORM_VOLUME_SHARPEN_ENTRIES,"uvCacheSharpenCells","uvCacheSharpenFaces"] as const) {
        const id = `uniform.volume.tiled.${entryPoint}`; ids.push(id);
        tasks.push({ id, phase: "solver-pipelines", label: `${entryPoint} (4h work map)`, run: async () => {
          (this.sharpenTilePipelines as Record<string,GPUComputePipeline>)[entryPoint] = await compiler.compileComputePipeline({
            label: `Uniform Geometric 4h - ${entryPoint}`, layout: this.mainPipelineLayout,
            compute: { module: shaderModule, entryPoint, constants: { [UNIFORM_VOLUME_TILE_WORK_OVERRIDE]: 1 } },
          }, { priority: "visible", signal });
        } });
      }
      const id = "uniform.volume.classify-tiles"; ids.push(id);
      tasks.push({ id, phase: "solver-pipelines", label: "Classify 4h sharpening work", run: async () => {
        this.tileClassifyPipeline = await compiler.compileComputePipeline({
          label: "Uniform Geometric 4h sharpening map", layout: this.mainPipelineLayout,
          compute: { module: shaderModule, entryPoint: UNIFORM_VOLUME_TILE_CLASSIFY_ENTRY },
        }, { priority: "visible", signal });
      } });
    }
    // E1's four map passes are always resident so the experiment is a live
    // toggle; with it off they are never encoded.
    if (this.twoLevelTileCount > 0) for (const entryPoint of UNIFORM_VOLUME_TWO_LEVEL_ENTRIES) {
      const id = `uniform.volume.twolevel.${entryPoint}`; ids.push(id);
      tasks.push({ id, phase: "solver-pipelines", label: entryPoint, run: async () => {
        this.twoLevelPipelines[entryPoint] = await compiler.compileComputePipeline({
          label: `Uniform Geometric E1 - ${entryPoint}`, layout: this.mainPipelineLayout,
          compute: { module: shaderModule, entryPoint },
        }, { priority: "visible", signal });
      } });
    }
    if (this.surfaceVolumeCorrection) {
      const id="uniform.volume.total-surface"; ids.push(id);
      tasks.push({id,phase:"solver-pipelines",label:"Total surface volume correction",
        run:()=>this.surfaceVolumeCorrection!.initialize(signal)});
    }
    if(this.fieldPages) { const id="uniform.fields.pages"; ids.push(id);
      tasks.push({id,phase:"solver-pipelines",label:"Uniform page publication",run:()=>this.fieldPages!.initialize(signal)}); }
    if(this.pageDomainPublication) { const id="uniform.domain.publication"; ids.push(id);
      tasks.push({id,phase:"solver-pipelines",label:"Accepted page domain publication",
        run:()=>this.pageDomainPublication!.initialize(signal)}); }
    const pipelineReadyId = "uniform.pipeline.publish";
    tasks.push({ id: pipelineReadyId, phase: "solver-pipelines", label: "Publish uniform reference programs", dependencies: ids, run: () => {
      this.pipelines = compiled as UniformReferencePipelines;
    } });
    const extrapolationReadyId = "uniform.extrapolation.pipelines";
    tasks.push({
      id: extrapolationReadyId,
      phase: "solver-pipelines",
      label: "Compile paper Sec. 3.3 extrapolation programs",
      run: () => this.velocityExtrapolator.initialize(signal),
    });
    const multigridReadyId = "uniform.pressure.multigrid";
    tasks.push({
      id: multigridReadyId,
      phase: "solver-pipelines",
      label: "Compile CM11a LCP multigrid programs",
      run: async () => {
        await this.pressureMultigrid.initialize({
          uniformBindGroupLayout: this.pressureInputLayout,
          shaderSource: `${this.shaderSource}\n${this.pressureMultigrid.shaderFragment}`,
          signal,
        });
        this.publishUniformPipelineFacts();
      },
    });
    tasks.push({ id: "uniform.surface.initial", phase: "upload", label: "Reconstruct smooth t=0 surface", dependencies: [pipelineReadyId, extrapolationReadyId, multigridReadyId], run: () => { this.info.allocatedBytes += (this.fieldPages?.allocationOverheadBytes ?? 0) + (this.surfaceVolumeCorrection?.allocatedBytes ?? 0); this.encodeInitialPresentationSurface(); } });
    tasks.push({ id: "uniform.warmup", phase: "warmup", label: "Fence uniform t=0 uploads", dependencies: ["uniform.surface.initial"], run: async () => { await this.device.queue.onSubmittedWorkDone(); } });
    return tasks;
  }

  private encodeInitialPresentationSurface(): void {
    if (!this.pipelines) throw new Error("Uniform reference pipelines are not initialized");
    this.writeParams(0, Math.min(this.scene.rigidBodies.length, 12), 0);
    const encoder = this.device.createCommandEncoder({ label: "Uniform reference t=0 surface reconstruction" });
    this.pageDomainPublication?.encode(encoder);
    // Seed static rho'/face-open authority once across the lattice. Runtime
    // updates only need the rolling active box; untouched air keeps this exact
    // static boundary state instead of paying the same full pass every frame.
    this.runDirect(encoder, "Uniform initial Sec. 3.3 interface authority",
      this.pipelines.extrapolationAuthorityDense, this.extrapolationAuthorityGroup,
      [Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4)]);
    if (this.geometricVolume) {
      // Use the same page traversal as runtime publication. A dense-shaped
      // launch cannot cover a page-shaped entry point on a large domain.
      this.run(encoder, "Uniform Geometric initial page surface publication",
        this.volumePipelines.uvPublish!, this.wallFilmResolveGroup);
    } else if (this.densityPostProcessing) {
      this.run(encoder, "Uniform initial post-process blur x", this.pipelines.postprocessBlurX, this.postprocessBlurXGroup);
      this.run(encoder, "Uniform initial post-process blur y", this.pipelines.postprocessBlurY, this.postprocessBlurYGroup);
      this.run(encoder, "Uniform initial post-process blur z", this.pipelines.postprocessBlurZ, this.postprocessBlurZGroup);
      this.run(encoder, "Uniform initial sub-grid surface resolve", this.pipelines.postprocessResolve, this.postprocessResolveGroup);
    } else {
      this.run(encoder, "Uniform initial wall-film resolve", this.pipelines.wallFilmResolve, this.wallFilmResolveGroup);
    }
    this.fieldPages?.encodePublications(encoder);
    this.device.queue.submit([encoder.finish()]);
  }

  private writeParams(dt: number, activeBodyCount: number, inflowStrength: number, drop?: InjectedLiquidBall): void {
    const c = this.scene.container;
    const inflow = this.scene.fluid.inflow;
    const outlet = this.inflowBoundary?.outletCenter_m;
    this.device.queue.writeBuffer(this.params, 0, new Float32Array([
      this.info.nx, this.info.ny, this.info.nz, dt,
      c.width_m / this.info.nx, c.height_m / this.info.ny, c.depth_m / this.info.nz, this.scene.fluid.gravity_m_s2.y,
      c.width_m, c.height_m, c.depth_m, sceneHasTerrain(this.scene) ? 1 : 0,
      this.scene.fluid.density_kg_m3, this.scene.fluid.dynamicViscosity_Pa_s,
      // E1's fine reach in 4h tiles, or -1 with the experiment off. Negative is
      // the whole gate: the two-level branch in sampleVelocityComponent is then
      // never taken and the four map passes are never encoded.
      // w: geometric only, V claims pressure rows. The paper shader's use of
      // this word (level-set authority) has always been written as zero.
      this.twoLevelEnabled ? this.twoLevelFineReach : -1, this.geometricVolume ? this.volumePressureRows : 0,
      this.scene.fluid.surfaceTension_N_m, c.fluidWallMode === "no-slip" ? 1 : 0, activeBodyCount, c.top === "open" ? 1 : 0,
      outlet?.x ?? 0, outlet?.y ?? 0, outlet?.z ?? 0, inflow?.radius_m ?? 0,
      inflow?.velocity_m_s.x ?? 0, inflow?.velocity_m_s.y ?? 0, inflow?.velocity_m_s.z ?? 0, this.inflowBoundary?.apertureScale ?? 0,
      inflowStrength, this.referenceVolumeCells, c.fillFraction * this.info.ny, 4,
      this.sharpeningStrength, this.sharpeningDistance,
      this.volumeDustThreshold,
      c.depthBoundary === "symmetry" ? 1 : 0,
      drop?.centre_m.x ?? 0, drop?.centre_m.y ?? 0, drop?.centre_m.z ?? 0, drop?.radius_m ?? 0,
      drop?.halfHeight_m ?? 0, this.liquidOnlyVelocityAdvection ? 1 : 0,
      // w: velocityD still holds this advance's V_face authority when pressure
      // runs. MacCormack and the stage audit both overwrite it after the store.
      this.solidVoxelScratchOffsetWords, this.sharedFaceOpenValid() ? 1 : 0,
      // The shell reach in 4h tiles, whether the extension's finest passes run
      // on those tiles, and whether advection and projection take their far-air
      // arm outside the fine tiles. All inert while the sampler is off.
      this.twoLevelShellReach, this.twoLevelExtensionEnabled ? 1 : 0,
      // w is E3's transport reach in 4h tiles past the fine set, or -1 with the
      // live set off -- the same negative gate physical.z uses, so every
      // transport kernel's tile test folds to "never skip" and the dense arm
      // executes the identical instruction stream it did before E3.
      this.twoLevelAdvectionEnabled ? 1 : 0,
      this.transportTilesEnabled ? this.transportReach + 8 : -1,
      // agreement: compaction, phi seed, shift gain and clamp. Geometric only.
      this.geometricVolume && this.volumeCompaction ? 1 : 0, this.geometricVolume && this.phiSeedFromVolume ? 1 : 0,
      this.geometricVolume ? this.phiAgreementGain : 0, this.phiAgreementClamp,
    ]));
  }

  /**
   * Add a ball of liquid to the solve that is already running.
   *
   * The alternative is re-seeding from the edited document, which throws away
   * the run the user is watching in order to add water to it — so authoring a
   * ball at t > 0 would mean losing everything that had happened. This is the
   * same mass source the nozzle uses, on the same guard, applied on exactly one
   * step: the ball appears in the field where it was dropped and is transported
   * from there like any other liquid.
   *
   * Consumed by the next step, which is also what happens if the clock is
   * paused when it is called — the ball lands when the clock next runs.
   */
  injectLiquidBall(ball: InjectedLiquidBall): void {
    if (!(ball.radius_m > 0)) return;
    this.pendingDrop = ball;
  }

  /** The editor's water shapes, as far as the one-step drop source reaches: a ball, added. */
  async editFluid(edit: LiveFluidEdit): Promise<LiveFluidEditResult> {
    if (edit.operation !== "add" || edit.shape !== "ball") {
      return { accepted: false, reason: "This fluid method adds water as balls only; choose Sparse Geometric to remove water or drop other shapes." };
    }
    const { width_m, height_m, depth_m } = this.scene.container, { nx, ny, nz } = this.info;
    const refusal = liveFluidEditRefusal(edit, { origin_m: [-width_m / 2, 0, -depth_m / 2],
      cellSize_m: [width_m / nx, height_m / ny, depth_m / nz], dimensions: [nx, ny, nz] });
    if (refusal) return { accepted: false, reason: refusal };
    if (this.pendingDrop) return { accepted: false, reason: "The previous drop lands when the clock next runs; run the simulation first." };
    this.injectLiquidBall({ centre_m: edit.center_m, radius_m: edit.radius_m });
    return { accepted: true };
  }

  /**
   * Adopt controls whose pipelines and storage are already resident.
   *
   * The renderer calls this once per resolved frame. Keep it allocation-free
   * and idempotent: stage gates alter host-side command encoding, while the
   * sharpening scalars reach WGSL through the existing per-advance params
   * buffer. Turning Sec. 3.8 on while paused is the one transition that needs
   * work immediately, so reconstruct the current density once for display.
   */
  applyRuntimeValues(values: MethodParamValues): void {
    this.faceAuthorityStored = false;
    // The renderer reapplies the latest values before the next advance.
    const finite = (key: string, fallback: number, minimum: number, maximum: number) => {
      const value = Number(values[key]);
      return Number.isFinite(value) ? Math.min(maximum, Math.max(minimum, value)) : fallback;
    };
    this.pressureMultigrid.setResidualTolerance(finite("pressureResidualTolerance", 10, 0, 100));
    // Switching to "fixed" mid-run restores the full encoded schedule on the
    // next step; switching back drops the previous demand sample and uses
    // the startup budget.
    const lagged = values.pressureCycleBudget !== "fixed";
    if (lagged !== this.pressureCycleBudgetLagged) this.pressureCyclesExecutedSample = undefined;
    this.pressureCycleBudgetLagged = lagged;
    this.pressureBudgetHeadroom = Math.round(finite(
      "pressureBudgetHeadroom", this.pageDomain ? 0 : UNIFORM_CM11A_DEFAULT_BUDGET_HEADROOM, 0, 4));
    const postProcessing = uniformDensityPostProcessingEnabled(
      values.densityPostProcessing,
      this.scene.sceneId,
    );
    const refreshPresentation = postProcessing !== this.densityPostProcessing;
    this.densityPostProcessing = postProcessing;
    this.densitySharpening = values.densitySharpening !== "off";
    this.sharpeningMassCorrection = values.sharpeningMassCorrection !== "off";
    this.gammaDiffusionIterations = values.gammaDiffusion === "off" ? 0 : Math.round(finite(
      "gammaDiffusionIterations", UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS, 1,
      UNIFORM_GAMMA_DIFFUSION_MAX_ITERATIONS,
    ));
    this.sharpeningStrength = finite("sharpeningStrength", 1, this.geometricVolume ? 0 : 0.25, this.geometricVolume ? 1 : 2);
    this.sharpeningDistance = finite("sharpeningDistance", 2.1, 0.1, 3.1);
    this.solidExcessCorrection = values.solidExcessCorrection !== "off";
    this.rigidCoupling = values.rigidCoupling !== "off";
    this.paperTimeStep = values.timeStep !== "scene";
    if(values.velocityTransport === "maccormack" && this.velocityC.width===1 && this.info.nx>1)
      throw new Error("Changing velocity transport requires rebuilding the Uniform Geometric solver");
    this.velocityTransport = values.velocityTransport === "maccormack"
      ? "maccormack" : "semi-lagrangian";
    this.liquidOnlyVelocityAdvection = values.liquidOnlyVelocityAdvection === "on";
    if (values.pressureWindow !== undefined) {
      const wanted = values.pressureWindow !== "domain" && this.geometricVolume
        && this.activeRegionEnabled && !this.windowDispatchIndirect;
      if (wanted !== this.pressureWindowLattice) {
        // Live, because the instance cache makes it live: the capacity this
        // step needs is either already built or is built synchronously from
        // the programs the load-time instance compiled. The shared level table
        // has to change hands at the same moment, because with the lattice on
        // it describes the extension's pyramid and with it off the pressure's.
        this.pressureWindowLattice = wanted;
        this.pressureWindowShrinkStreak = 0;
        this.pressureWindowDomainSteps = UNIFORM_WINDOW_LAG_STEPS;
        this.adoptPressureInstance([this.info.nx, this.info.ny, this.info.nz], [0, 0, 0]);
        this.writeActiveLevelDimensions();
      }
    }
    // Partial-values callers must not reset the sweep budget.
    if (values.extensionFrontSweeps !== undefined && Number(values.extensionFrontSweeps) !== this.velocityExtrapolator.frontPasses) {
      this.velocityExtrapolator.setFrontPasses(Number(values.extensionFrontSweeps));
      this.publishUniformPipelineFacts();
    }
    if (this.geometricVolume) {
      // Absent means "leave as constructed": partial value records must not
      // silently switch a dense-constructed solver onto the work map.
      if (values.sharpeningWorkMap !== undefined) this.geometricTileWork = values.sharpeningWorkMap !== "off";
      if (values.volumeDustThreshold !== undefined) this.volumeDustThreshold = finite("volumeDustThreshold", 0, 0, 1);
      if (values.twoLevelVelocity !== undefined) this.twoLevelVelocity = this.twoLevelTileCount > 0 && values.twoLevelVelocity === "on";
      if (values.twoLevelFineReach !== undefined) this.twoLevelFineReach = Math.round(finite("twoLevelFineReach", 2, 0, 8));
      if (values.twoLevelExtension !== undefined) this.twoLevelExtensionTiles = values.twoLevelExtension !== "dense";
      if (values.twoLevelAdvection !== undefined) this.twoLevelAdvectionTiles = values.twoLevelAdvection !== "dense";
      if (values.twoLevelShellReach !== undefined) this.twoLevelShellReach = Math.round(finite("twoLevelShellReach", 1, 0, 8));
      if (values.transportWorkMap !== undefined) this.transportTiles = values.transportWorkMap !== "dense";
      if (values.transportReach !== undefined) this.transportReach = Math.round(finite("transportReach", 1, -8, 8));
      if (values.volumePressureRows !== undefined) this.volumePressureRows = values.volumePressureRows === "all" ? 2 : values.volumePressureRows === "off" ? 0 : 1;
      if (values.surfaceDeficitBalancing !== undefined) this.surfaceDeficitBalancing = !!this.surfaceVolumeCorrection && values.surfaceDeficitBalancing === "on";
      if (values.totalSurfaceVolume !== undefined) this.totalSurfaceVolume = values.totalSurfaceVolume === "on";
      if (values.volumeCompaction !== undefined) this.volumeCompaction = values.volumeCompaction === "on";
      if (values.phiSeedFromVolume !== undefined) this.phiSeedFromVolume = values.phiSeedFromVolume === "on";
      if (values.phiAgreement !== undefined) this.phiAgreementGain = values.phiAgreement === "on" ? finite("phiAgreementGain", 0.05, 0, 1) : 0;
      if (values.phiAgreementClamp !== undefined) this.phiAgreementClamp = finite("phiAgreementClamp", 0.02, 0, 0.5);
      this.geometricRedistance = values.redistance !== "off";
      this.solidExcessCorrection = false;
      this.densityPostProcessing = false;
      this.gammaDiffusionIterations = 0;
    }
    if (refreshPresentation && this.pipelines) {
      // The refresh rewrites the represented surface the owed reduction reads.
      const encoder = this.device.createCommandEncoder({ label: "Uniform reference owed diagnostics reduction" });
      if (this.encodeOwedDiagnosticsReduction(encoder)) this.device.queue.submit([encoder.finish()]);
      this.encodeInitialPresentationSurface();
    }
  }

  /** Pays the reduction the last step skipped, from the fields it left behind. */
  private encodeOwedDiagnosticsReduction(encoder: GPUCommandEncoder): boolean {
    if (!this.diagnosticsReductionOwed || !this.pipelines) return false;
    this.diagnosticsReductionOwed = false;
    this.run(encoder, "Uniform diagnostics reduction", this.pipelines.reduce, this.reductionGroup);
    return true;
  }

  private initializeVolumeAndTerrain(): void {
    const { nx, ny, nz } = this.info;
    const c = this.scene.container;
    const {volume,terrain,initial,wetMinimum,wetMaximum,dam} = uniformInitialVolume(this.scene,[nx,ny,nz],this.geometricVolume);
    const cellHeight = c.height_m / ny;
    if (this.vertexPhiField && this.vertexPhiScratch) {
      const phi = uniformVolumeInitialPhi(this.scene, [nx, ny, nz]);
      this.upload3DF32(this.vertexPhiField, phi, nx+1, ny+1, nz+1);
      this.upload3DF32(this.vertexPhiScratch, phi, nx+1, ny+1, nz+1);
    }
    this.upload3DF32(this.volumeA, volume, nx, ny, nz);
    this.upload3DF32(this.volumeB, volume, nx, ny, nz);
    this.upload3DF32(this.surfaceA, volume, nx, ny, nz);
    this.upload3DF32(this.surfaceB, volume, nx, ny, nz);
    const gamma = new Float32Array(nx * ny * nz).fill(1);
    this.upload3DF32(this.gammaA, gamma, nx, ny, nz);
    this.upload3DF32(this.gammaB, gamma, nx, ny, nz);
    const zeroBoundaryVelocity = new Float32Array(this.negativeBoundaryVelocityBytes / 4);
    this.device.queue.writeBuffer(this.boundaryVelocityA, 0, zeroBoundaryVelocity);
    this.device.queue.writeBuffer(this.boundaryVelocityB, 0, zeroBoundaryVelocity);
    this.device.queue.writeBuffer(this.boundaryVelocityC, 0, zeroBoundaryVelocity);
    this.device.queue.writeBuffer(this.boundaryVelocityD, 0, zeroBoundaryVelocity);
    this.referenceVolumeCells = initial;
    const terrainCells = Float32Array.from(terrain, (height) => height / cellHeight);
    this.upload2DF32(this.terrainTexture, terrainCells, nx, nz);
    this.initializeActiveRegion(wetMinimum, wetMaximum);
    Object.assign(this.info, {
      initialVolumeCellSum: initial, volumeCellSum: initial,
      representedVolumeCellSum: initial, volumeDrift: 0, representedVolumeDrift: 0,
      rawVolumeDrift: 0, volumeTelemetrySource: "initial-condition",
      maxSpeed_m_s: 0,
      front_m: this.scene.fluid.initialCondition === "dam-break"
        ? -c.width_m / 2 + dam.max.x * c.width_m : c.width_m / 2,
      frontTelemetrySource: "initial-condition",
    });
  }

  private initializeActiveRegion(wetMinimum: number[], wetMaximum: number[]): void {
    const dimensions = [this.info.nx, this.info.ny, this.info.nz];
    const words = new Uint32Array(UNIFORM_ACTIVE_HEADER_WORDS);
    const empty = wetMaximum.every((value) => value === 0);
    // Uniform Geometric seeds the window at the WHOLE domain and lets the
    // first step's scan shrink it, which costs two conservative steps. The
    // t=0 publication is a dense dispatch, but every geometric kernel adds the
    // window origin to its id, so a t=0 window that did not start at the
    // origin would leave the static open-fraction plane unwritten below it --
    // for a droplet in a tall tank, that is most of the domain. From the first
    // step on, the GPU finalize keeps the geometric window's origin a multiple
    // of four, so a 4^3 workgroup is still exactly one 4h tile.
    const wholeDomain = !this.activeRegionEnabled || this.geometricVolume;
    const minimum = wholeDomain || empty ? [0, 0, 0]
      : wetMinimum.map((value) => Math.max(0, value - 8));
    const maximum = wholeDomain ? dimensions
      : empty ? [1, 1, 1]
      : wetMaximum.map((value, axis) => Math.min(dimensions[axis]!, value + 8));
    // Words 0..5 retain the padded wet/source box from the previous step.
    // Words 7..12 are the union of two consecutive boxes, giving ping-pong
    // targets one clearing tail without retaining the entire swept history.
    words.set(minimum, 0);
    words.set(maximum, 3);
    words.set(minimum, 7);
    words.set(maximum, 10);
    words.set(maximum.map((value, axis) => Math.ceil((value - minimum[axis]!) / 4)), 13);
    words.set(maximum.map((value, axis) => Math.ceil((value - minimum[axis]! + 1) / 4)),
      UNIFORM_ACTIVE_VERTEX_DISPATCH_WORD);
    this.activeLevelDimensions.forEach((level, index) => {
      if (index >= UNIFORM_ACTIVE_MAX_LEVELS) return;
      const base = UNIFORM_ACTIVE_LEVEL_BASE_WORD + index * UNIFORM_ACTIVE_LEVEL_WORDS;
      const scale = dimensions.map((value, axis) => value / level[axis]!);
      const origin = minimum.map((value, axis) => Math.max(0, Math.floor(value / scale[axis]!) - 2));
      const end = maximum.map((value, axis) => Math.min(level[axis]! + 2,
        Math.ceil(value / scale[axis]!) + 3));
      words.set(origin, base);
      words.set(end.map((value, axis) => Math.ceil((value - origin[axis]!) / 4)), base + 3);
      words.set(level, base + 6);
    });
    this.device.queue.writeBuffer(this.activeRegion, 0, words);
    if(this.phiRegion)this.device.queue.writeBuffer(this.phiRegion,0,words);
    this.device.queue.writeBuffer(this.activeScratch, 0, words);
    this.device.queue.writeBuffer(this.activeDispatch, 0, words);
    // A reset re-seeds the box, so every lagged host count is stale: the next
    // steps run whole-domain counts until a fresh readback lands.
    this.windowLagged = undefined;
    this.windowMainGroups = undefined;
    this.windowVertexGroups = undefined;
    this.windowViolations = 0;
    this.windowViolationAxes = 0;
    this.pressureMultigrid.setWindowLevelGroups(undefined);
    this.velocityExtrapolator.setWindowGroups(undefined, undefined);
    // The CPU-known starting box, so the opening steps can plan a window
    // lattice instead of building the domain-sized instance nothing reuses.
    this.pressureWindowSeedBox = empty ? undefined
      : { minimum: [...wetMinimum], maximum: [...wetMaximum] };
    // A reset seeds the geometric window at the whole domain, so the first
    // step's union box is the whole domain too and no smaller lattice could
    // contain it. Two steps of domain capacity cover that and the step after.
    this.pressureWindowDomainSteps = UNIFORM_WINDOW_LAG_STEPS;
    this.pressureWindowShrinkStreak = 0;
    this.adoptPressureInstance([...dimensions] as [number, number, number], [0, 0, 0]);
  }

  private copyField(encoder: GPUCommandEncoder, source: GPUImageCopyTexture, destination: GPUImageCopyTexture, size: GPUExtent3D): void {
    if(this.fieldPages) this.fieldPages.copy(encoder,source.texture,destination.texture);
    else encoder.copyTextureToTexture(source,destination,size);
  }

  /** Initialization hook for manufactured GPU boundary fixtures. Public textures
   * are read-only presentation adapters when fields use page storage. */
  initializeVelocityForQA(values: Float32Array): void {
    if(this.lastTime!==0 || values.length!==4*this.info.cellCount)throw new Error("Velocity fixture must initialize the complete field at t=0");
    this.upload3DF32(this.velocityA,values,this.info.nx,this.info.ny,this.info.nz);
  }

  private upload3DF32(texture: GPUTexture, values: Float32Array, nx: number, ny: number, nz: number): void {
    if(this.fieldPages) { this.fieldPages.upload(texture,values); return; }
    const components=texture.format==="rgba32float"?4:1;
    const rowBytes = nx * components * 4, padded = Math.ceil(rowBytes / 256) * 256;
    const packed = new Uint8Array(padded * ny * nz), source = new Uint8Array(values.buffer);
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      const row = y + ny * z;
      packed.set(source.subarray(rowBytes * row, rowBytes * (row + 1)), padded * row);
    }
    this.device.queue.writeTexture({ texture }, packed, { bytesPerRow: padded, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
  }

  private upload2DF32(texture: GPUTexture, values: Float32Array, nx: number, nz: number): void {
    const rowBytes = nx * 4, padded = Math.ceil(rowBytes / 256) * 256;
    const packed = new Uint8Array(padded * nz), source = new Uint8Array(values.buffer);
    for (let z = 0; z < nz; z += 1) packed.set(source.subarray(rowBytes * z, rowBytes * (z + 1)), padded * z);
    this.device.queue.writeTexture({ texture }, packed, { bytesPerRow: padded, rowsPerImage: nz }, { width: nx, height: nz });
  }

  private dispatch(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup): void {
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    if(this.pageDomainDispatch){pass.dispatchWorkgroupsIndirect(this.pageDomainDispatch,0);}
    else if (this.windowMainGroups) {
      pass.dispatchWorkgroups(...this.windowMainGroups);
    } else if (this.activeRegionEnabled) {
      pass.dispatchWorkgroupsIndirect(this.activeDispatch, UNIFORM_ACTIVE_MAIN_DISPATCH_OFFSET);
    } else pass.dispatchWorkgroups(
      Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
  }

  private sharedFaceOpenValid(): boolean {
    return this.geometricVolume && uniformAbOn("facecache") && !this.activeRegionEnabled
      && this.velocityTransport !== "maccormack" && !this.symmetryStageAuditFields;
  }

  private run(encoder: GPUCommandEncoder, label: string, pipeline: GPUComputePipeline, group: GPUBindGroup): void {
    const pass = encoder.beginComputePass({ label });
    this.dispatch(pass, pipeline, group);
    pass.end();
  }

  private runDirect(encoder: GPUCommandEncoder, label: string, pipeline: GPUComputePipeline,
    group: GPUBindGroup, workgroups: readonly [number, number, number]): void {
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(...workgroups); pass.end();
  }

  /**
   * Uniform Geometric's (n+1)^3 vertex lattice, windowed.
   *
   * The window's own origin serves the vertex passes unchanged — vertex v
   * belongs to cells v-1..v — so only the extent differs, by one vertex per
   * axis. `finalizeActiveRegion` publishes that record; with the window off
   * this is the dense (n+1)/4 dispatch it always was.
   */
  private runVertex(encoder: GPUCommandEncoder, label: string, pipeline: GPUComputePipeline,
    group: GPUBindGroup): void {
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline); pass.setBindGroup(0, this.phiGroups.get(group) ?? group);
    if(this.phiDispatch){pass.dispatchWorkgroupsIndirect(this.phiDispatch,UNIFORM_ACTIVE_VERTEX_DISPATCH_OFFSET);}
    else if(this.phiShaderSource){pass.dispatchWorkgroups(Math.ceil((this.info.nx+1)/4),Math.ceil((this.info.ny+1)/4),Math.ceil((this.info.nz+1)/4));}
    else if(this.pageDomainDispatch){pass.dispatchWorkgroupsIndirect(this.pageDomainDispatch,16);}
    else if (this.windowVertexGroups) {
      pass.dispatchWorkgroups(...this.windowVertexGroups);
    } else if (this.activeRegionEnabled) {
      pass.dispatchWorkgroupsIndirect(this.activeDispatch, UNIFORM_ACTIVE_VERTEX_DISPATCH_OFFSET);
    } else pass.dispatchWorkgroups(Math.ceil((this.info.nx + 1) / 4),
      Math.ceil((this.info.ny + 1) / 4), Math.ceil((this.info.nz + 1) / 4));
    pass.end();
  }

  /**
   * Size this step's dispatches on the host, from the box the GPU published a
   * couple of steps ago.
   *
   * The window is exact where it has to be and lagged where it is cheap: every
   * kernel still reads the ORIGIN (and each level's origin) out of
   * `activeRegion`, written by this step's own scan/reduce/finalize, so no
   * sample moves. Only the group COUNT is chosen here, and a count is only
   * ever wrong by being too small -- threads past the exact extent exit
   * immediately in `activeId`, so the slack costs launches and nothing else.
   *
   * The count must therefore cover however much the box can have grown since
   * the readback: `UNIFORM_WINDOW_LAG_STEPS` steps of the per-axis two-sided
   * travel the GPU measured, plus a cell per side and the four-cell tile
   * alignment. `finalizeActiveRegion` checks the exact extent
   * against what was chosen and counts every step it did not fit; one such
   * step clips the far edge of the window (the front stalls there; V is
   * neither created nor destroyed, since an undispatched cell keeps the value
   * it already had) and buys `UNIFORM_WINDOW_VIOLATION_PENALTY_STEPS` steps of
   * whole-domain counts.
   *
   * Dense counts are also forced before the first readback, after a reset or
   * scene edit, and when an external source introduces new support, because on those the
   * lagged box says nothing about where liquid is about to be.
   */
  private planWindowDispatch(dt: number, externalSources: boolean): void {
    if (!this.activeRegionEnabled || this.windowDispatchIndirect) return;
    const dims = [this.info.nx, this.info.ny, this.info.nz];
    const container = this.scene.container;
    const spacing = [container.width_m / dims[0]!, container.height_m / dims[1]!,
      container.depth_m / dims[2]!];
    this.windowStep += 1;
    const lagged = this.windowLagged;
    if (lagged) {
      this.windowMaxLagSteps = Math.max(this.windowMaxLagSteps,
        this.windowStep - this.windowLaggedStep);
    }
    // A source or a scene edit invalidates the lagged box for as long as the
    // readback takes to catch up, not just for this step.
    if (externalSources || this.activeRegionRescanPending) {
      this.windowForcedDenseSteps = Math.max(this.windowForcedDenseSteps,
        UNIFORM_WINDOW_LAG_STEPS + 1);
      // Liquid can appear anywhere on these steps, so the pressure lattice
      // goes back to the domain until the readback describes where it landed.
      this.pressureWindowDomainSteps = Math.max(this.pressureWindowDomainSteps,
        UNIFORM_WINDOW_LAG_STEPS + 1);
    }
    const dense = lagged === undefined || this.windowForcedDenseSteps > 0;
    if (this.windowForcedDenseSteps > 0) this.windowForcedDenseSteps -= 1;
    if (dense) this.windowDenseSteps += 1;
    this.planPressureLattice(dense, lagged);
    // Verification hook: FLUID_UNIFORM_WINDOW_LAG_PAD=0 starves the lag
    // allowance so the exact box outgrows the host's counts on purpose, which
    // is how the clipped-step path is shown to stall a front rather than
    // create or destroy volume.
    const padOverride = typeof process !== "undefined"
      ? Number(process.env.FLUID_UNIFORM_WINDOW_LAG_PAD) : Number.NaN;
    const extent = dims.map((n, axis) => {
      if (dense) return n;
      // The exact box grows by at most this step's two-sided travel per step,
      // which the GPU measured per axis over the wet cells and published. Two
      // steps of that, a cell per side for the rounding, and four more for the
      // growth of the exact PADDING itself: the padding carries the same
      // travel term, so a step that accelerates widens the box twice over.
      const travel = lagged!.travel[axis] ?? Math.ceil(
        (2 * lagged!.speed_m_s * dt) / spacing[axis]!);
      const lag = Number.isFinite(padOverride) ? padOverride
        : UNIFORM_WINDOW_LAG_STEPS * (travel + 2) + 4;
      const padded = lagged!.maximum[axis]! - lagged!.minimum[axis]! + lag;
      return Math.min(n, Math.ceil(padded / 4) * 4);
    });
    const groups = (values: readonly number[]): [number, number, number] =>
      [Math.ceil(values[0]! / 4), Math.ceil(values[1]! / 4), Math.ceil(values[2]! / 4)];
    this.windowMainGroups = groups(extent);
    // The phi passes run on the cell box dilated by their own internal read
    // reach on both sides (VERTEX_PHI_REACH in the shader), so every phi a phi
    // pass reads was written this step. The host's counts must be a superset
    // of that box, and `extent` is already a superset of the exact span.
    this.windowVertexGroups = groups(extent.map((value) =>
      value + 1 + 2 * UNIFORM_VERTEX_PHI_REACH));
    // Per level, the same walk `finalizeActiveRegion` does: the level box is
    // the scaled cell box, grown by the two-cell low halo and the three-cell
    // high one, and one more for the floor/ceil pair the scaling can straddle.
    const levelGroups = this.activeLevelDimensions.map((level) =>
      groups(level.map((size, axis) =>
        Math.min(size + 2, Math.ceil(extent[axis]! * size / dims[axis]!) + 6))));
    const base = UNIFORM_ACTIVE_CPU_LEVEL_BASE_WORD;
    const words = new Uint32Array(UNIFORM_ACTIVE_CPU_MODE_WORD + 1 - base);
    levelGroups.forEach((level, index) => {
      if (index >= UNIFORM_ACTIVE_MAX_LEVELS) return;
      words.set(level, index * 3);
    });
    words.set(this.windowMainGroups, UNIFORM_ACTIVE_CPU_MAIN_WORD - base);
    words.set(this.windowVertexGroups, UNIFORM_ACTIVE_CPU_VERTEX_WORD - base);
    words[UNIFORM_ACTIVE_CPU_MODE_WORD - base] = dense ? 0 : 1;
    this.device.queue.writeBuffer(this.activeScratch, base * 4, words);
    this.pressureMultigrid.setWindowLevelGroups(levelGroups);
    this.velocityExtrapolator.setWindowGroups(this.windowMainGroups, levelGroups);
  }

  /**
   * Level dimensions the shared active-region table describes.
   *
   * Two consumers read that table: the pressure hierarchy's own per-level
   * dispatches and the extension's fill pyramid, which maps record = its level
   * + 1. They can only share one table while both hierarchies halve the same
   * way. With the pressure lattice planned on the window the pressure passes
   * stop consulting the table altogether, so it is seeded from the extension's
   * own pyramid instead -- which also fixes the 8x case, where the pressure
   * plan semi-coarsens and the records described a hierarchy the extension
   * does not have.
   */
  private get activeLevelDimensions(): readonly (readonly [number, number, number])[] {
    if (!this.pressureWindowLattice) return this.pressureMultigrid.levelPhysicalDimensions;
    return [[this.info.nx, this.info.ny, this.info.nz] as const,
      ...this.velocityExtrapolator.hierarchyLevelDimensions];
  }

  /**
   * Choose this step's window-local CM11a lattice: capacity, origin, instance.
   *
   * The lattice must contain every cell the window works on, so it is sized
   * from the same lagged box and the same per-side travel the dispatch counts
   * use, and `finalizeActiveRegion` checks the exact box against it and counts
   * the steps it did not cover. Capacity is aligned so coarse grids stay
   * registered to the domain, which is what keeps the origin still while the
   * liquid moves inside it.
   *
   * Growth is immediate; shrinking waits, because a capacity change is the
   * only expensive thing here and a surface that breathes across an alignment
   * boundary would otherwise re-plan every few steps.
   */
  private planPressureLattice(dense: boolean,
    lagged: { minimum: number[]; maximum: number[]; travel: number[] } | undefined): void {
    const domain: [number, number, number] = [this.info.nx, this.info.ny, this.info.nz];
    const wholeDomain = !this.pressureWindowLattice || this.pressureWindowDomainSteps > 0;
    if (this.pressureWindowDomainSteps > 0) this.pressureWindowDomainSteps -= 1;
    if (wholeDomain) {
      // A domain-capacity instance at origin zero is bit-identical under the
      // window rule: every halo's simulation coordinate leaves the domain, so
      // every halo is the wall or lid it always was. The mode word therefore
      // stays on whenever the lattice is enabled at all, and only the capacity
      // and origin move -- there is no second code path to keep in step.
      this.adoptPressureInstance(domain, [0, 0, 0]);
      // The domain instance carries these steps, but the window will want its
      // own capacity the moment the lag expires -- at start-up, and again after
      // every violation. Build it here rather than paying the walk on the step
      // that switches.
      if (this.pressureWindowLattice) {
        // From the SEED box, not the lagged one: a reset seeds the geometric
        // window at the whole domain, so the readback these steps see is the
        // domain itself and would prewarm the capacity already in hand. The
        // CPU-known starting box is what the window will actually be.
        const { low, high } = this.pressureLatticeBox(domain, undefined);
        this.prewarmCapacity(planUniformCM11aWindow(domain, low, high).capacity);
      }
      return;
    }
    const { low, high, travel } = this.pressureLatticeBox(domain, lagged);
    const target = planUniformCM11aWindow(domain, low, high);
    const seated = seatUniformCM11aWindow(domain, this.pressureWindowCapacity,
      target.alignment, low, high);
    // The hysteresis exists to stop a surface breathing across an alignment
    // boundary from re-planning every few steps. Leaving the WHOLE DOMAIN is
    // not that: it is the initial seating, and the state every violation and
    // every reset returns to. Waiting thirty steps for it would make the
    // startup plan -- the one the window exists to replace -- the cost of the
    // first second of every scene, and of every recovery after one.
    const fromDomain = this.pressureWindowCapacity.join("x") === this.pressureDomainKey;
    if (!fromDomain && seated
      && !this.pressureInstances.get(this.pressureWindowCapacity.join("x"))?.destroyed) {
      const smaller = target.capacity.reduce((product, value) => product * value, 1)
        < this.pressureWindowCapacity.reduce((product, value) => product * value, 1);
      this.pressureWindowShrinkStreak = smaller ? this.pressureWindowShrinkStreak + 1 : 0;
      if (this.pressureWindowShrinkStreak < UNIFORM_PRESSURE_WINDOW_SHRINK_STEPS) {
        this.adoptPressureInstance(this.pressureWindowCapacity, seated);
        this.prewarmNeighbourLattice(domain, low, high, target.alignment, travel);
        return;
      }
    }
    this.pressureWindowShrinkStreak = 0;
    this.adoptPressureInstance([...target.capacity] as [number, number, number],
      [...target.origin] as [number, number, number]);
    this.prewarmNeighbourLattice(domain, low, high, target.alignment, travel);
    void dense;
  }

  /**
   * The padded box the pressure lattice must cover this step.
   *
   * Before the first readback the host still knows where the liquid started,
   * so the opening steps plan from the seed box rather than the domain.
   */
  private pressureLatticeBox(domain: [number, number, number],
    lagged: { minimum: number[]; maximum: number[]; travel: number[] } | undefined):
    { low: [number, number, number]; high: [number, number, number]; travel: readonly number[] } {
    const box = lagged ?? this.pressureWindowSeedBox;
    const travel = lagged?.travel ?? [0, 0, 0];
    const low: [number, number, number] = [0, 0, 0];
    const high: [number, number, number] = [0, 0, 0];
    for (let axis = 0; axis < 3; axis += 1) {
      const slack = lagged
        ? UNIFORM_WINDOW_LAG_STEPS * (travel[axis] ?? 0) + 4
        : UNIFORM_PRESSURE_WINDOW_SEED_PAD;
      low[axis] = Math.max(0, (box?.minimum[axis] ?? 0) - slack);
      high[axis] = Math.min(domain[axis]!, (box?.maximum[axis] ?? domain[axis]!) + slack);
    }
    return { low, high, travel };
  }

  /**
   * Build the capacity the window is about to need, a little each step.
   *
   * The trigger is headroom, not growth rate: once the padded box comes within
   * eight cells of filling the capacity on the axis with the least room, the
   * next step up on that axis is planned in the background. Nothing here
   * touches the current instance or the encoded step -- the prewarm is an
   * unreferenced lattice until `adoptPressureInstance` asks for its key -- so
   * the worst case of a wrong guess is the memory it holds and the host time
   * it spent, both of which stop as soon as the headroom recovers.
   */
  private prewarmNeighbourLattice(domain: [number, number, number],
    low: readonly number[], high: readonly number[], alignment: readonly number[],
    travel: readonly number[]): void {
    let axis = -1;
    let tightest = Number.POSITIVE_INFINITY;
    for (let candidate = 0; candidate < 3; candidate += 1) {
      // An axis already at the domain has nowhere to grow.
      if (this.pressureWindowCapacity[candidate]! >= domain[candidate]!) continue;
      const headroom = this.pressureWindowCapacity[candidate]!
        - (high[candidate]! - low[candidate]!);
      const lead = UNIFORM_PRESSURE_PREWARM_HEADROOM
        + UNIFORM_PRESSURE_PREWARM_STEPS * (travel[candidate] ?? 0);
      if (headroom <= lead && headroom < tightest) { tightest = headroom; axis = candidate; }
    }
    if (axis < 0) { this.discardPressurePrewarm(); return; }
    const grown = [...high];
    grown[axis] = Math.min(domain[axis]!, high[axis]! + alignment[axis]!);
    this.prewarmCapacity(planUniformCM11aWindow(domain, low as [number, number, number],
      grown as [number, number, number]).capacity);
  }

  /** Create or advance the prewarm for one capacity; see the caller above. */
  private prewarmCapacity(capacity: readonly number[]): void {
    const key = capacity.join("x");
    if (key === this.pressureWindowCapacity.join("x") || this.pressureInstances.has(key)) {
      this.discardPressurePrewarm(); return;
    }
    if (this.pressurePrewarm && this.pressurePrewarm.key !== key) this.discardPressurePrewarm();
    const started = typeof performance !== "undefined" ? performance.now() : 0;
    if (!this.pressurePrewarm) {
      const programs = this.pressureMultigrid.programs;
      if (!programs) return;
      try {
        this.pressurePrewarm = { key, instance: new WebGPUUniformPressureMultigrid(this.device,
          [...capacity] as [number, number, number], [
            this.scene.container.width_m / this.info.nx,
            this.scene.container.height_m / this.info.ny,
            this.scene.container.depth_m / this.info.nz,
          ], this.pressureSchedule, this.activeRegionEnabled ? this.activeDispatch : undefined,
          programs, true) };
        this.pressurePrewarm.instance.setResidualTolerance(this.pressureMultigrid.residualTolerance);
      } catch {
        // A capacity the planner cannot build is not worth a step; the switch
        // will fall back to the domain instance as it always did.
        this.pressurePrewarm = undefined; return;
      }
      // The step that allocates the lattice does no walking: allocation is the
      // one part of a build that cannot be cut into cycles, so it gets the
      // step to itself.
    } else {
      // One cycle at a time until the budget is spent. The check is after a
      // cycle, not inside one, so a step overshoots by at most the cycle it
      // was in the middle of; a cycle is the smallest resumable unit the plan
      // walk has, because everything finer shares the ping-pong parities.
      this.pressurePrewarm.instance.advancePlan(UNIFORM_PRESSURE_PREWARM_BUDGET_MS);
    }
    this.pressurePrewarmMs += (typeof performance !== "undefined" ? performance.now() : 0) - started;
    this.pressurePrewarmSteps += 1;
  }

  private discardPressurePrewarm(): void {
    this.pressurePrewarm?.instance.destroy();
    this.pressurePrewarm = undefined;
  }

  /**
   * Make an instance of this capacity current, creating it if the cache has
   * none. Creation is synchronous: the compiled programs come from the
   * instance built at load, so nothing here compiles or awaits.
   */
  private adoptPressureInstance(capacity: [number, number, number],
    origin: [number, number, number]): void {
    const key = capacity.join("x");
    let instance = this.pressureInstances.get(key);
    if (!instance && this.pressurePrewarm?.key === key) {
      // The prewarm guessed right. Whatever cycles it has left cost a fraction
      // of the walk, and they are the only thing standing between here and a
      // ready instance.
      const started = typeof performance !== "undefined" ? performance.now() : 0;
      instance = this.pressurePrewarm.instance;
      instance.advancePlan(Number.POSITIVE_INFINITY);
      this.pressurePrewarm = undefined;
      this.pressureWindowReplanMs = (typeof performance !== "undefined" ? performance.now() : 0) - started;
      this.pressureWindowReplans += 1;
      this.pressureInstances.set(key, instance);
    }
    if (!instance) {
      const programs = this.pressureMultigrid.programs;
      const started = typeof performance !== "undefined" ? performance.now() : 0;
      try {
        if (!programs) throw new Error("Uniform CM11a programs are not compiled");
        instance = new WebGPUUniformPressureMultigrid(this.device, capacity, [
          this.scene.container.width_m / this.info.nx,
          this.scene.container.height_m / this.info.ny,
          this.scene.container.depth_m / this.info.nz,
        ], this.pressureSchedule, this.activeRegionEnabled ? this.activeDispatch : undefined,
          programs);
        instance.setResidualTolerance(this.pressureMultigrid.residualTolerance);
      } catch {
        // A capacity the planner cannot build is not worth failing a step for:
        // fall back to the domain instance, which always exists.
        this.pressureWindowDomainSteps = Math.max(this.pressureWindowDomainSteps,
          UNIFORM_WINDOW_VIOLATION_PENALTY_STEPS);
        if (key !== this.pressureDomainKey) {
          this.adoptPressureInstance([this.info.nx, this.info.ny, this.info.nz], [0, 0, 0]);
        }
        return;
      }
      this.pressureWindowReplanMs = (typeof performance !== "undefined" ? performance.now() : 0) - started;
      this.pressureWindowReplans += 1;
      this.pressureInstances.set(key, instance);
    } else {
      this.pressureInstances.delete(key); this.pressureInstances.set(key, instance);
    }
    for (const [evictKey, evicted] of [...this.pressureInstances]) {
      if (this.pressureInstances.size <= UNIFORM_PRESSURE_WINDOW_CACHE) break;
      if (evictKey === key || evictKey === this.pressureDomainKey) continue;
      evicted.destroy(); this.pressureInstances.delete(evictKey);
    }
    if (instance !== this.pressureMultigrid) {
      this.pressureMultigrid = instance;
      this.projectGroup = this.makeProjectGroup(instance.pressureTexture);
      // Level count, per-stage pass counts and the plan total all move with
      // the capacity, and the panel's chips read them from the info record.
      this.publishUniformPipelineFacts();
      if (this.pressureCoarsestCapture !== undefined) {
        instance.enableCoarsestCapture(this.pressureCoarsestCapture);
      }
    }
    instance.setWindowLattice(this.pressureWindowLattice);
    this.pressureWindowCapacity = capacity;
    this.pressureWindowOrigin = origin;
    this.pressureWindowLatticeActive = this.pressureWindowLattice
      && key !== this.pressureDomainKey;
    this.writePressureLatticeHeader(origin, capacity, this.pressureWindowLattice);
  }

  /**
   * Rewrite the physical dimensions in the shared level table.
   *
   * `finalizeActiveRegion` derives every level's origin and group count from
   * these words each step, so flipping which hierarchy the table describes is
   * this one write plus the next finalize.
   */
  private writeActiveLevelDimensions(): void {
    const words = new Uint32Array(UNIFORM_ACTIVE_MAX_LEVELS * UNIFORM_ACTIVE_LEVEL_WORDS);
    this.activeLevelDimensions.forEach((level, index) => {
      if (index >= UNIFORM_ACTIVE_MAX_LEVELS) return;
      words.set(level, index * UNIFORM_ACTIVE_LEVEL_WORDS + 6);
    });
    // Only the dimension triple of each record: origins and group counts are
    // the finalize's, and the stride keeps them where they were.
    for (let index = 0; index < UNIFORM_ACTIVE_MAX_LEVELS; index += 1) {
      const base = (UNIFORM_ACTIVE_LEVEL_BASE_WORD + index * UNIFORM_ACTIVE_LEVEL_WORDS + 6) * 4;
      const triple = words.slice(index * UNIFORM_ACTIVE_LEVEL_WORDS + 6,
        index * UNIFORM_ACTIVE_LEVEL_WORDS + 9);
      this.device.queue.writeBuffer(this.activeScratch, base, triple);
      this.device.queue.writeBuffer(this.activeRegion, base, triple);
    }
  }

  /** Publish the lattice the GPU must agree with: origin, capacity, mode. */
  private writePressureLatticeHeader(origin: readonly number[], capacity: readonly number[],
    windowLattice: boolean): void {
    const words = this.pressureWindowHeaderWords;
    words.set(origin, 0); words.set(capacity, 3); words[6] = windowLattice ? 1 : 0;
    this.device.queue.writeBuffer(this.activeScratch,
      UNIFORM_ACTIVE_PRESSURE_ORIGIN_WORD * 4, words);
    this.device.queue.writeBuffer(this.activeRegion,
      UNIFORM_ACTIVE_PRESSURE_ORIGIN_WORD * 4, words);
  }

  /**
   * Ask the queue for the box this step published, without waiting for it.
   *
   * Same shape as `readPressureCycleDemand`: its own small buffer, nothing in
   * the frame path awaits it, and a step simply skips the copy while an
   * earlier map is outstanding. It never gates correctness -- only how tightly
   * the NEXT steps size their dispatches.
   */
  private readWindowBox(): void {
    const buffer = this.windowReadback;
    if (!buffer || this.disposed) return;
    this.windowReadbackPending = true;
    void buffer.mapAsync(GPUMapMode.READ).then(() => {
      if (this.disposed) { this.windowReadbackPending = false; return; }
      try {
        const words = new Uint32Array(buffer.getMappedRange().slice(0));
        const travelBits = words[18]!;
        this.windowLagged = {
          minimum: [words[7]!, words[8]!, words[9]!],
          maximum: [words[10]!, words[11]!, words[12]!],
          speed_m_s: new Float32Array(new Uint32Array([words[6]!]).buffer)[0]!,
          travel: [travelBits & 1023, (travelBits >>> 10) & 1023, (travelBits >>> 20) & 1023],
        };
        this.windowLaggedStep = this.windowReadbackStep;
        const violations = words[16]!;
        if (violations > this.windowViolations) {
          this.windowForcedDenseSteps = UNIFORM_WINDOW_VIOLATION_PENALTY_STEPS;
          this.pressureWindowDomainSteps = UNIFORM_WINDOW_VIOLATION_PENALTY_STEPS;
          this.windowViolationAxes = words[17]!;
        }
        this.windowViolations = violations;
      } finally {
        if (buffer.mapState === "mapped") buffer.unmap();
        this.windowReadbackPending = false;
      }
    }).catch(() => { this.windowReadbackPending = false; });
  }

  private encodePhiRegion(encoder: GPUCommandEncoder): void {
    if(this.phiRegion){
      // Live dust-floor changes invalidate the zero-volume-outside predicate.
      // The page-domain header is the prebuilt full-domain fallback.
      if(this.volumeDustThreshold<=0){
        encoder.copyBufferToBuffer(this.activeRegion,0,this.phiRegion,0,UNIFORM_ACTIVE_HEADER_WORDS*4);
        encoder.copyBufferToBuffer(this.activeRegion,0,this.phiDispatch!,0,UNIFORM_ACTIVE_HEADER_WORDS*4);
        return;
      }
      const group=this.phiGroups.get(this.reductionGroup)!;
      this.runDirect(encoder,"Phi support census",this.pipelines!.scanExternalActiveSources,group,
        [Math.ceil(this.info.nx/4),Math.ceil(this.info.ny/4),Math.ceil(this.info.nz/4)]);
      this.runDirect(encoder,"Phi support reduction",this.pipelines!.reduceExternalActiveRegionSummaries,group,[1,1,1]);
      this.runDirect(encoder,"Phi support closure",this.pipelines!.finalizeActiveRegion,group,[1,1,1]);
      encoder.copyBufferToBuffer(this.activeScratch,0,this.phiRegion,0,UNIFORM_ACTIVE_HEADER_WORDS*4);
      encoder.copyBufferToBuffer(this.activeScratch,0,this.phiDispatch!,0,UNIFORM_ACTIVE_HEADER_WORDS*4);
    }
  }

  private encodeActiveRegion(encoder: GPUCommandEncoder, externalSources: boolean): void {
    if (!this.pipelines) throw new Error("Uniform reference pipelines are not initialized");
    if (!this.activeRegionEnabled) return;
    // A live scene edit can move a solid into liquid, or change the terrain,
    // anywhere in the domain. One dense census restores a conservative box.
    const forced = this.activeRegionRescanPending;
    this.activeRegionRescanPending = false;
    if (externalSources || forced) {
      // A new inlet/drop may begin beyond the previous box. One dense pass
      // summarizes both existing liquid and sources, avoiding a second census.
      this.runDirect(encoder, "Uniform scan liquid and external active sources",
        this.pipelines.scanExternalActiveSources, this.reductionGroup,
        [Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4)]);
      this.runDirect(encoder, "Uniform reduce external-source workgroup summaries",
        this.pipelines.reduceExternalActiveRegionSummaries, this.reductionGroup, [1, 1, 1]);
    } else {
      this.run(encoder, "Uniform scan prior active liquid bounds", this.pipelines.scanActiveRegion,
        this.reductionGroup);
      this.runDirect(encoder, "Uniform reduce active workgroup summaries",
        this.pipelines.reduceActiveRegionSummaries, this.reductionGroup, [1, 1, 1]);
    }
    this.runDirect(encoder, "Uniform finalize active liquid dispatches", this.pipelines.finalizeActiveRegion,
      this.reductionGroup, [1, 1, 1]);
    encoder.copyBufferToBuffer(this.activeScratch, 0, this.activeRegion, 0,
      UNIFORM_ACTIVE_HEADER_WORDS * 4);
    encoder.copyBufferToBuffer(this.activeScratch, 0, this.activeDispatch, 0,
      UNIFORM_ACTIVE_HEADER_WORDS * 4);
  }

  private encodeVelocityExtrapolation(
    encoder: GPUCommandEncoder,
    predicted: boolean,
    seam?: (phase: GPUTimestampPhase) => void,
  ): void {
    if (!this.pipelines) throw new Error("Uniform reference pipelines are not initialized");
    // Only a full-lattice store may be trusted later: a windowed one leaves
    // cells outside it holding whatever geometry they last saw.
    const faceAuthorityStable = uniformAbOn("authoritystatic") && this.stepBodyCount === 0
      && this.velocityTransport !== "maccormack" && !this.symmetryStageAuditFields
      && !this.pageDomainDispatch && !this.windowMainGroups && !this.activeRegionEnabled;
    if (faceAuthorityStable && this.faceAuthorityStored) this.run(encoder, "Uniform Sec. 3.3 rho-prime authority",
      this.pipelines.extrapolationDensityAuthority, this.extrapolationAuthorityGroup);
    else this.run(encoder, "Uniform Sec. 3.3 rho-prime and face authority",
      this.pipelines.extrapolationAuthority, this.extrapolationAuthorityGroup);
    this.faceAuthorityStored = faceAuthorityStable;
    if (!predicted && this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.surfaceA },
      { texture: this.symmetryStageAuditFields.extrapolationDensityAuthority },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (!predicted) seam?.(UNIFORM_ADVANCE_PHASE.extensionAuthority);
    // The 4h field the two-level sampler reads is this hierarchy's own
    // ceil(n/4) level, published inside the fill phase once it has completed.
    // MacCormack extends twice per step, so the table always describes the
    // field the next sample would otherwise have interpolated finely.
    this.velocityExtrapolator.encode(encoder, predicted, !predicted && seam ? ((stage) => seam(
      stage === "narrow-band-front" ? UNIFORM_ADVANCE_PHASE.extensionFront
        : UNIFORM_ADVANCE_PHASE.extensionHierarchy,
    )) : undefined, this.twoLevelEncoded);
  }

  /**
   * Instance-accurate pass counts for the pipeline panel's chips. Plain data on
   * the info record so it survives the worker structured-clone boundary; the
   * counts depend on grid dimensions, so they publish once the multigrid plan
   * and extrapolation hierarchy exist.
   */
  private publishUniformPipelineFacts(): void {
    const multigridPasses = this.pressureMultigrid.planStageCounts;
    if (!multigridPasses) return;
    this.info.uniformPipelineFacts = {
      extrapolationFrontSweeps: this.velocityExtrapolator.frontPasses,
      extrapolationHierarchyLevels: this.velocityExtrapolator.hierarchyLevelCount,
      extrapolationPassesPerInvocation: this.velocityExtrapolator.encodedPassCount,
      multigridLevels: this.pressureMultigrid.levelCount,
      multigridPasses,
      multigridPassesTotal: Object.values(multigridPasses).reduce((sum, count) => sum + count, 0),
      pressureSchedule: this.pressureSchedule,
    };
  }

  /**
   * Cycles the next pressure solve encodes, and the telemetry that explains it.
   *
   * Fixed mode returns the configured schedule, so its command stream is the
   * one the solver encoded before P1 existed. Lagged mode sizes the encoded
   * prefix from the latest asynchronous diagnostics sample; the GPU-side
   * residual gate still stops a converged solve inside that prefix.
   */
  private planPressureCycleBudget(): number {
    const maxCycles = this.pressureMultigrid.cycleCount;
    // A zero tolerance means "run every configured cycle": the GPU gate never
    // stops, so the executed count reports the schedule rather than the
    // demand, and lagging on it would silently cap a solve the operator asked
    // to run in full. The budget stands down whenever the gate is disabled.
    const lagged = this.pressureCycleBudgetLagged
      && this.pressureMultigrid.residualTolerance > 0;
    const budget = lagged
      ? uniformCM11aCycleBudget({
        lastExecutedCycles: this.pressureCyclesExecutedSample,
        initialCycles: this.pageDomain ? UNIFORM_CM11A_MINIMUM_CYCLE_BUDGET : undefined,
        lastConverged: this.pressureCycleConvergedSample,
        headroom: this.pressureBudgetHeadroom,
        minCycles: UNIFORM_CM11A_MINIMUM_CYCLE_BUDGET,
        maxCycles,
      })
      : maxCycles;
    Object.assign(this.info, {
      hostSchedulingUsesReadback: lagged,
      uniformPressureCycleBudget: lagged ? "lagged" : "fixed",
      uniformPressureBudgetHeadroom: this.pressureCycleBudgetLagged
        ? this.pressureBudgetHeadroom : undefined,
      uniformPressureCyclesEncoded: budget,
      uniformPressureCyclesConfigured: maxCycles,
      uniformPressurePassesEncoded: this.pressureMultigrid.encodedPassCount(budget),
      uniformPressurePassesConfigured: this.pressureMultigrid.planPassCount,
    });
    return budget;
  }

  /**
   * Ask the queue for the cycle counters this step just wrote.
   *
   * This is the same asynchronous, post-submit readback shape as `readStats`,
   * on its own twelve-byte buffer so the two never contend: nothing in the
   * frame path waits on it, and a step simply skips the copy while an earlier
   * map is still outstanding. It never gates correctness — only how many
   * cycles the *next* step bothers to encode.
   */
  private readPressureCycleDemand(): void {
    const buffer = this.pressureCycleDemandReadback;
    if (!buffer || this.disposed) return;
    this.pressureCycleDemandPending = true;
    void buffer.mapAsync(GPUMapMode.READ).then(() => {
      if (this.disposed) { this.pressureCycleDemandPending = false; return; }
      try {
        const words = new Uint32Array(buffer.getMappedRange().slice(0));
        this.pressureCycleConvergedSample = words[0] === 1;
        this.pressureCyclesExecutedSample = words[1]! + words[2]!;
        // Word 22 of the state block: a cycle was rejected and the finish ran.
        this.pressureRecoveryExpected = words[6] !== 0;
        Object.assign(this.info, {
          uniformPressureCyclesExecuted: this.pressureCyclesExecutedSample,
          uniformPressureCyclesConverged: this.pressureCycleConvergedSample,
        });
      } finally {
        if (buffer.mapState === "mapped") buffer.unmap();
        this.pressureCycleDemandPending = false;
      }
    }).catch(() => { this.pressureCycleDemandPending = false; });
  }

  /** The 4h map is requested, dimensionally possible, and sharpening is on. */
  private get sharpenTileWork(): boolean {
    return this.geometricTileWork && this.sharpenTileCount > 0 && this.densitySharpening;
  }

  /**
   * The tile classes the most recent step ran on. Withdrawn while the sampler is
   * off, when the records go stale and every cell reads the finest lattice.
   */
  get tileClassSource(): GPUFluidTileClassSource | undefined {
    return this.twoLevelEncoded ? this.tileClassRecords : undefined;
  }

  /**
   * The active-region header this step's finalize wrote, for the solve-window
   * view. Withdrawn while the window is off, when the dense schedule dispatches
   * the whole domain and the header is only its t=0 seed.
   */
  get solveWindowSource(): GPUFluidSolveWindowSource | undefined {
    return this.activeRegionEnabled ? { records: { buffer: this.activeRegion } } : undefined;
  }

  /** Experiment E1 is requested, dimensionally possible, and geometric. */
  private get twoLevelEnabled(): boolean {
    return this.twoLevelVelocity && this.twoLevelTileCount > 0;
  }

  /** E2 shrinks the extension only under the sampler that replaces its output. */
  private get twoLevelExtensionEnabled(): boolean {
    return this.twoLevelEnabled && this.twoLevelExtensionTiles;
  }

  /** E2b needs the same map, so it is gated on the same sampler. */
  private get twoLevelAdvectionEnabled(): boolean {
    return this.twoLevelEnabled && this.twoLevelAdvectionTiles;
  }

  /**
   * E3 needs the class map E1 builds, and it needs its own predicate to hold.
   * The gather writes zero outside the live set, so a cell outside it that
   * still carried V would simply lose it. With the dust floor at zero a cell
   * can carry ULP-scale V anywhere in the domain and "V is zero outside the
   * live set" fails, so the dense schedule is forced instead.
   */
  private get transportTilesEnabled(): boolean {
    return this.twoLevelEnabled && this.transportTiles && this.volumeDustThreshold > 0;
  }

  /** Byte offset of the shell-tile counter, above the two dilation planes. */
  private get twoLevelShellCountOffset(): number {
    return ((this.scratchArena ? 0 : this.info.nx * this.info.ny * this.info.nz) + 6 * this.twoLevelTileCount) * 4;
  }

  /** Byte offset of the classify dispatch's active-tile counter. */
  private get sharpenTileCountWordOffset(): number {
    return ((this.scratchArena?.sharpenBaseWords ?? 2 * this.info.nx * this.info.ny * this.info.nz)
      + UNIFORM_VOLUME_SHARPEN_TILE_COUNT_WORD) * 4;
  }

  private createPageAwareGroup(descriptor: GPUBindGroupDescriptor, scratchBytes?: number): GPUBindGroup {
    const group = this.fieldPages ? this.fieldPages.createBindGroup(descriptor,true,scratchBytes) : this.device.createBindGroup(descriptor);
    this.groupDescriptors.set(group,descriptor);
    return group;
  }

  get volumePageSource(): GPUFluidVolumePageSource | undefined {
    const p = this.volumePageConfig;
    const workRecords = p ? { buffer: this.conditioningScratch, offset: p.base * 4, size: (8 + 2 * p.count) * 4 } : undefined;
    const records = this.pageDomainView ? {buffer:this.pageDomainView} : workRecords;
    return records ? { records, workRecords,
      transportRecords: this.volumeTransportPageView ? {buffer:this.volumeTransportPageView} : undefined } : undefined;
  }

  /** GPU-only page assignment and compact work lists. The backing arena is
   * reserved at construction; no mapping, allocation or submission splits an
   * advance. Every record read by a phase is initialized by an earlier pass. */
  private prepareVolumePages(encoder: GPUCommandEncoder, sharpen: boolean): void {
    const p=this.volumePageConfig!;
    encoder.clearBuffer(this.conditioningScratch,4*(p.base+8),4*p.count);
    if(p.work)encoder.clearBuffer(this.conditioningScratch,4*(p.base+8+2*p.count),4);
    // Encode the phase flag as a GPU command. queue.writeBuffer would change
    // both phases before this single command buffer starts executing.
    if(sharpen && this.sharpenTileWork)
      encoder.copyBufferToBuffer(this.volumePageSharpenFlag!,0,this.conditioningScratch,4*(p.base+5),4);
    else encoder.clearBuffer(this.conditioningScratch,4*(p.base+5),4);
    const entry=sharpen?"uvMarkSharpenPages":"uvMarkTransportPages";
    this.run(encoder,entry,this.pagePipelines[entry]!,sharpen?this.sharpenComputeGroup:this.densityTraceGroup);
    this.runDirect(encoder,"Compact volume pages",this.pagePipelines.uvCompactPages!,this.densityTraceGroup,[1,1,1]);
    if(p.work){
      encoder.copyBufferToBuffer(this.conditioningScratch,4*(p.base+6),this.volumeWorkDispatch!,0,8);
      encoder.copyBufferToBuffer(this.conditioningScratch,4*(p.base+8+2*p.count),this.volumeWorkCounts!,sharpen?4:0,4);
    }
    // Preserve the transport set before sharpening reuses its flags. Metadata
    // only: no field copies, readback or additional numerical dispatch.
    if (!sharpen) encoder.copyBufferToBuffer(this.conditioningScratch,4*p.base,this.volumeTransportPageView!,0,4*(8+2*p.count));
    this.info.uniformVolumePageStage=sharpen?"sharpening":"transport";
  }

  private runVolumeWork(encoder:GPUCommandEncoder,label:string,pipeline:GPUComputePipeline,group:GPUBindGroup):void {
    if(!this.volumeWorkDispatch){this.run(encoder,label,pipeline,group);return;}
    const pass=encoder.beginComputePass({label});pass.setPipeline(pipeline);
    pass.setBindGroup(0,group);
    pass.dispatchWorkgroupsIndirect(this.volumeWorkDispatch,0);pass.end();
  }

  private sharpenPipeline(entry: typeof UNIFORM_VOLUME_SHARPEN_ENTRIES[number]): GPUComputePipeline {
    return (this.sharpenTileWork ? this.sharpenTilePipelines[entry] : undefined)
      ?? this.volumePipelines[entry]!;
  }

  /** Two GPU reductions, using the post-transport surface target already in gammaA. */
  private encodeSurfaceDeficitBalance(encoder: GPUCommandEncoder): void {
    if (!this.geometricVolume) return;
    if (!this.surfaceDeficitBalancing) {
      // Clear the previous rate when toggled off; never reuse a stale source.
      encoder.clearBuffer(this.conditioningScratch, (this.scratchArena?.conditioningBytes ?? this.info.cellCount * 12), 4);
      return;
    }
    // Clear the count even for an empty accepted generation. Partial records
    // follow page traversal, including padded lanes of partial boundary pages.
    encoder.clearBuffer(this.conditioningScratch, (this.scratchArena?.conditioningBytes ?? this.info.cellCount * 12) + 4, 4);
    if (this.pageDomain) this.run(encoder, "Surface-deficit page sums",
      this.volumePipelines.uvBalanceMeasure!, this.sharpenComputeGroup);
    else this.runDirect(encoder, "Surface-deficit partial sums", this.volumePipelines.uvBalanceMeasure!,
      this.sharpenComputeGroup, [Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4)]);
    this.runDirect(encoder, "Surface-deficit global balance", this.volumePipelines.uvBalanceReduce!, this.sharpenComputeGroup, [1, 1, 1]);
  }

  private encodeGeometricVolume(encoder: GPUCommandEncoder, seam?: (phase: GPUTimestampPhase) => void): void {
    const run = (entry: typeof UNIFORM_VOLUME_ENTRIES[number], group = this.densityTraceGroup) => {
      if(entry === "uvFinishDonorSums") {
        if(this.pageDomain) this.run(encoder,entry,this.volumePipelines[entry]!,this.volumeDonorGroup);
        else this.runDirect(encoder,entry,this.volumePipelines[entry]!,this.volumeDonorGroup,[Math.ceil(this.info.nx/4),Math.ceil(this.info.ny/4),Math.ceil(this.info.nz/4)]);
      } else if(entry==="uvBuildEdges"||entry==="uvFallback"||entry==="uvNormalizeRows"||entry==="uvNormalizeDonors")
        this.runVolumeWork(encoder,entry,this.volumePipelines[entry]!, (this.scratchArena || entry==="uvBuildEdges"||entry==="uvNormalizeRows")?this.volumeDonorGroup:group);
      else this.run(encoder,entry,this.volumePipelines[entry]!,group);
    };

    // The shift's residual is packed into the gamma scratch half from
    // start-of-step V, gamma and phi, and the advect then binds that half as
    // its gamma input. uvGather and uvPublish both rewrite it later this step.
    const shift = this.phiAgreementGain > 0;
    if (shift) run("uvAgreementResidual");
    this.runVertex(encoder, "Advect page vertex phi", this.volumePipelines.uvAdvectPhi!, shift ? this.densityGatherGroup : this.densityTraceGroup);
    if (this.geometricRedistance) this.runVertex(encoder, "Redistance page vertex phi", this.volumePipelines.uvRedistancePhi!, this.phiReverseGroup!);
    else this.copyField(encoder,{texture:this.vertexPhiScratch!},{texture:this.vertexPhiField!},[this.info.nx+1,this.info.ny+1,this.info.nz+1]);
    seam?.(UNIFORM_VOLUME_PHASE.phi);
    // Deposit donor weights while each row is already in registers. Integer
    // accumulation is exact and order independent, so this saves four full
    // edge-table scans without changing the transport normalization scheme.
    if (this.volumePageConfig) this.prepareVolumePages(encoder,false);
    encoder.clearBuffer(this.volumeDonorSums!,this.scratchArena?.donorOffset ?? 0,this.scratchArena?.donorBytes);
    run("uvBuildEdges"); run("uvFinishDonorSums"); run("uvFallback");
    for (let round = 0; round < 3; round++) {
      encoder.clearBuffer(this.volumeDonorSums!,this.scratchArena?.donorOffset ?? 0,this.scratchArena?.donorBytes);
      run("uvNormalizeRows"); run("uvFinishDonorSums"); run("uvNormalizeDonors");
    }
    seam?.(UNIFORM_VOLUME_PHASE.coupling);
    run("uvGather");
    // Whole-domain reduction: a window-local total would silently lose the
    // contribution of sleeping liquid. Capacity and target refreshes are dense.
    // Gather has already added hose/drop volume, so its current V is the
    // correction target on source steps too. Continuous inflow must not
    // silently disable the user's total-surface-volume constraint.
    if (this.totalSurfaceVolume && this.surfaceVolumeCorrection) {
      const dense=[Math.ceil(this.info.nx/4),Math.ceil(this.info.ny/4),Math.ceil(this.info.nz/4)] as const;
      this.runDirect(encoder,"Surface volume capacities",this.volumePipelines.uvCorrectionCapacity!,this.densityTraceGroup,dense);
      const priorBytes=this.surfaceVolumeCorrection.allocatedBytes;
      this.surfaceVolumeCorrection.encode(encoder);
      this.info.allocatedBytes+=this.surfaceVolumeCorrection.allocatedBytes-priorBytes;
      this.runDirect(encoder,"Refresh corrected surface targets",this.volumePipelines.uvCorrectionTargets!,this.densityTraceGroup,dense);
    }
    this.copyField(encoder,{texture:this.gammaB},{texture:this.gammaA},[this.info.nx,this.info.ny,this.info.nz]);
    seam?.(UNIFORM_VOLUME_PHASE.gather);
    this.sharpenTileMapEncoded = this.sharpenTileWork;
    if (this.sharpenTileMapEncoded) {
      encoder.clearBuffer(this.conditioningScratch, this.sharpenTileCountWordOffset, 4);
      this.run(encoder, "Classify 4h sharpening work", this.tileClassifyPipeline!, this.sharpenComputeGroup);
    }
    if (this.volumePageConfig && this.densitySharpening) this.prepareVolumePages(encoder,true);
    // Inactive sharpening tiles are identity through all eight sweeps. Seed
    // the other ping-pong half once instead of copying them on every commit.
    if(this.volumeWorkDispatch && this.densitySharpening)
      this.copyField(encoder,{texture:this.volumeB},{texture:this.volumeA},[this.info.nx,this.info.ny,this.info.nz]);
    if(this.volumeWorkDispatch && this.densitySharpening){
      this.runVolumeWork(encoder,"Cache sharpening cell geometry",(this.sharpenTileWork?(this.sharpenTilePipelines as Record<string,GPUComputePipeline>).uvCacheSharpenCells:this.volumePipelines.uvCacheSharpenCells)!,this.sharpenComputeGroup);
      this.runVolumeWork(encoder,"Cache sharpening face geometry",(this.sharpenTileWork?(this.sharpenTilePipelines as Record<string,GPUComputePipeline>).uvCacheSharpenFaces:this.volumePipelines.uvCacheSharpenFaces)!,this.sharpenComputeGroup);
    }
    if (this.densitySharpening) for (let round = 0; round < 8; round++) {
      const group = round % 2 === 0 ? this.sharpenComputeGroup : this.sharpenResolveGroup;
      for (const entry of UNIFORM_VOLUME_SHARPEN_ENTRIES) this.runVolumeWork(encoder, entry, this.sharpenPipeline(entry), group);
    }
    if (this.densitySharpening) seam?.(UNIFORM_VOLUME_PHASE.sharpen);
  }

  get pressureSmoothingWorkSourceForQA() { return this.pressureMultigrid.smoothingWorkSource; }

  get framePending(): boolean { return false; }
  async awaitFrameCompletion(): Promise<void> { await this.device.queue.onSubmittedWorkDone(); }
  async assertSimulationHealthy(completion?: Promise<void>): Promise<void> {
    // Presentation already awaited this frame. A second queue-wide fence here
    // includes newer presentations and holds their predecessor's throughput slot.
    await (completion ?? this.awaitFrameCompletion());
  }

  advanceTo(time_s: number, bodies: RigidBodyState[] = []): boolean {
    if (this.disposed) return false;
    // The paper's method is calibrated for its own large-step regime (dt=1/30
    // in every Sec. 4 example): sharpening opposes per-resample transport
    // blur, so far smaller scene steps structurally out-diffuse it.
    // A paper advance is exactly 1/30 s. Treating that value only as maxDt
    // let browser callers feed 4 ms targets, producing a different 250 Hz
    // method than the Dawn/paper lane and overwhelming sharpening with many
    // extra resamples. Accumulate target time until one complete paper step
    // is available; never encode a fractional paper step.
    if (this.paperTimeStep && !uniformPaperAdvanceReady(time_s, this.lastTime)) return false;
    const advance = planGPUAdvance(time_s, this.lastTime,
      this.paperTimeStep ? UNIFORM_PAPER_DT_S : this.scene.numerics.maxDt_s);
    if (!advance) return false;
    if (!this.pipelines) throw new Error("Uniform reference pipelines are not initialized");
    const dt = advance.dt_s;
    this.lastTime = advance.nextTime_s;
    this.info.submittedTime_s = this.lastTime;
    this.info.simulatedTime_s = this.lastTime;
    this.info.simulationLag_s = advance.lag_s;
    this.info.lastDt_s = dt;
    this.info.lastSubsteps = 1;
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + 1;
    const activeBodies = bodies.slice(0, 12);
    this.stepBodyCount = activeBodies.length;
    this.rigidSystem.syncBodies(activeBodies);
    const c = this.scene.container;
    const inflow = this.scene.fluid.inflow;
    const strength = inflow ? averageInflowStrength(inflow, this.lastTime - dt, this.lastTime) : 0;
    if (this.inflowBoundary && strength > 0) {
      const cellVolume = c.width_m * c.height_m * c.depth_m
        / (this.info.nx * this.info.ny * this.info.nz);
      this.referenceVolumeCells += this.inflowBoundary.flowRate_m3_s * strength * dt / cellVolume;
    }
    // The drop is consumed by this step and only this step, so the lane is
    // cleared before anything can encode a second one. Its mass joins the
    // reference the same way the nozzle's does — analytically, from the volume
    // asked for rather than from the volume that fitted — because the guard
    // that caps a cell at full is the same guard, and the drift telemetry is
    // the instrument that says how much the two disagreed.
    const drop = this.pendingDrop;
    this.pendingDrop = undefined;
    if (drop) {
      const cellVolume = c.width_m * c.height_m * c.depth_m
        / (this.info.nx * this.info.ny * this.info.nz);
      const dropped = drop.halfHeight_m !== undefined
        ? Math.PI * drop.radius_m ** 2 * 2 * drop.halfHeight_m
        : (4 / 3) * Math.PI * drop.radius_m ** 3;
      this.referenceVolumeCells += dropped / cellVolume;
    }
    this.info.referenceLiquidVolume_cells = this.referenceVolumeCells;
    this.writeParams(dt, activeBodies.length, strength, drop);
    // The advance-pipeline trace: a hardware-timestamp boundary chain over the
    // whole step, sampled on a cadence while the instrumentation store asks for
    // it. Boundaries splice into the next real pass's timestampWrites, so an
    // instrumented advance encodes the same passes as an uninstrumented one.
    const instrumentation = usePerformanceInstrumentationStore.getState();
    const traceRequestedAt_ms = instrumentation.enabled ? performance.now() : 0;
    const shouldTracePhysics = instrumentation.enabled && !this.physicsTracePending
      && traceRequestedAt_ms - this.lastPhysicsTraceAt_ms >= UNIFORM_PHYSICS_TRACE_CADENCE_MS;
    const physicsTraceSampleId = shouldTracePhysics ? ++this.physicsTraceSampleId : 0;
    const physicsTraceContext = `uniform:sim-${this.lastTime.toFixed(6)}`;
    const physicsCPUTrace = shouldTracePhysics
      ? new CPUPerformanceTrace(physicsTraceSampleId, physicsTraceContext,
        { id: "command-encoding", label: "Uniform advance planning + command encoding" })
      : undefined;
    // markersReady: without the compiled closing marker the final boundary
    // decodes as unsampled and one bad sample would retire hardware tracing.
    // Until it resolves (constructor kicks it off) the queue-wall observation
    // covers the sample without latching the fallback.
    const physicsTrace = shouldTracePhysics && !this.hardwarePhysicsTraceInvalid
      && GPUStageTimestampRecorder.supported(this.device)
      && GPUStageTimestampRecorder.markersReady(this.device)
      ? new GPUStageTimestampRecorder(this.device, physicsTraceSampleId, "physics", physicsTraceContext)
      : undefined;
    const physicsQueueTrace = shouldTracePhysics
      ? new GPUQueueWallPerformanceTraceRecorder(physicsTraceSampleId, "physics", physicsTraceContext)
      : undefined;
    const rawEncoder = this.device.createCommandEncoder({ label: "Uniform reference step" });
    const encoder = physicsTrace ? physicsTrace.instrument(rawEncoder) : rawEncoder;
    physicsTrace?.begin();
    const seam = physicsTrace || physicsCPUTrace
      ? (phase: GPUTimestampPhase) => {
        physicsCPUTrace?.completePhase(phase);
        physicsTrace?.completePhase(encoder, phase);
      }
      : undefined;
    // Preserve Sec. 3.6 unplaceable-excess telemetry written during the step.
    encoder.clearBuffer(this.reductions);
    encoder.clearBuffer(this.rigidExchange);
    if (!this.nativeRootExecution) this.pageDomainPublication?.encode(encoder);
    this.encodePhiRegion(encoder);
    if(!this.pageDomain){
    // A continuing inlet is retained by the GPU window seed at full strength.
    // Its support only needs rediscovery on activation, a larger swept extent,
    // or a scene edit (activeRegionRescanPending). Drops remain arbitrary.
    const inflowSupportDt = Math.fround(dt); // Match the timestep uploaded to WGSL.
    const windowExternalSources = drop !== undefined
      || (strength > 0 && inflowSupportDt > this.inflowWindowDt);
    this.inflowWindowDt = strength > 0 ? inflowSupportDt : 0;
    this.planWindowDispatch(dt, windowExternalSources);
    // Keep the complete source census for now: remote near-interface phi can
    // remain outside the previous window, especially after live insertion.
    // Census coverage must not force every later kernel to use domain counts.
    this.encodeActiveRegion(encoder, strength > 0 || drop !== undefined);
    }
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.velocityA }, { texture: this.symmetryStageAuditFields.preExtrapolationVelocity },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.volumeA }, { texture: this.symmetryStageAuditFields.previousRawDensity },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (this.symmetryStageAuditNegativeBoundaryVelocity) encoder.copyBufferToBuffer(
      this.boundaryVelocityA, 0, this.symmetryStageAuditNegativeBoundaryVelocity, 0,
      this.negativeBoundaryVelocityBytes,
    );
    this.twoLevelEncoded = this.twoLevelEnabled;
    this.transportTilesEncoded = this.transportTilesEnabled;
    // The tile classes, at the HEAD of the step and before the extension that
    // now runs on them: seeded from start-of-step V and phi, solids and this
    // step's sources, then dilated to FINE (k tiles) and SHELL (k + shell
    // reach) in one separated three-axis scan. Nothing between here and the
    // transport stage writes V or phi, so the classes a post-extension pass
    // would have produced are the same ones; and nothing between here and the
    // sharpening map clears words [N,2N), which is where they live.
    if (this.twoLevelEncoded) {
      const tiles: [number, number, number] = [Math.ceil(this.info.nx/4), Math.ceil(this.info.ny/4), Math.ceil(this.info.nz/4)];
      const grid: [number, number, number] = [Math.ceil(tiles[0]/4), Math.ceil(tiles[1]/4), Math.ceil(tiles[2]/4)];
      // Shell tiles, transport tiles and the measured maximum displacement.
      encoder.clearBuffer(this.conditioningScratch, this.twoLevelShellCountOffset,
        UNIFORM_VOLUME_TWO_LEVEL_COUNTER_WORDS * 4);
      for (const entry of UNIFORM_VOLUME_TWO_LEVEL_ENTRIES) {
        this.runDirect(encoder, `Uniform Geometric two-level ${entry}`, this.twoLevelPipelines[entry]!, this.densityTraceGroup, grid);
      }
    }
    this.encodeVelocityExtrapolation(encoder, false, seam);

    // Sec. 3.6 must see the updated solid geometry before Sec. 3.4 excludes
    // solid donors. Without this pair, density in cells newly covered by a
    // moving body is skipped by beta construction and then overwritten with
    // zero by the gather, so the supposedly conservative operator loses the
    // displaced liquid before the historical post-sharpening cleanup runs.
    if (this.solidExcessCorrection && (activeBodies.length > 0 || sceneHasTerrain(this.scene))) {
      // Ranged to the donor-sum region for the same reason the transport
      // clears are: the scatter/resolve pair only addresses [0,N), and words
      // [N,2N) carry the 4h classes for the rest of the step. (Uniform
      // Geometric forces this stage off; the range keeps it safe if it returns.)
      encoder.clearBuffer(this.conditioningScratch, 0, this.info.nx * this.info.ny * this.info.nz * 4);
      this.run(encoder, "Uniform moving-solid entry excess scatter",
        this.pipelines.scatterSolidExcess, this.solidEntryScatterGroup);
      this.run(encoder, "Uniform moving-solid entry excess resolve",
        this.pipelines.resolveSolidExcess, this.solidEntryResolveGroup);
    }

    if (this.geometricVolume) {
      this.encodeGeometricVolume(encoder, seam);
    } else {
    // Algorithm 1 steps 1-2, paper Secs. 3.3-3.5: use the extrapolated
    // current velocity for the modified conservative semi-Lagrangian density
    // operator, diffuse gamma in each dimension, then sharpen locally.
    encoder.clearBuffer(this.conditioningScratch);
    this.run(encoder, "Uniform trace gamma and beta", this.pipelines.traceGammaBeta, this.densityTraceGroup);
    if (this.symmetryStageAuditBetaBuffer) encoder.copyBufferToBuffer(
      this.conditioningScratch, 0, this.symmetryStageAuditBetaBuffer, 0,
      this.info.nx * this.info.ny * this.info.nz * 4,
    );
    this.run(encoder, "Uniform scatter density deficits", this.pipelines.scatterDensityDeficit, this.densityScatterGroup);
    this.run(encoder, "Uniform gather conservative density", this.pipelines.gatherDensity, this.densityGatherGroup);
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.volumeB }, { texture: this.symmetryStageAuditFields.densityAdvection },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.gammaA }, { texture: this.symmetryStageAuditFields.gammaPostAdvection },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    seam?.(UNIFORM_ADVANCE_PHASE.densityAdvection);
    // Sec. 3.4 step 8 and LAF11 Sec. 3.2: Jacobi within a dimension,
    // Gauss-Seidel between dimensions. Each dispatch gathers both face fluxes
    // from one immutable input snapshot; x -> y -> z ping-pongs those complete
    // snapshots. This is three dispatches per paper repetition. Splitting a
    // dimension into even/odd pair passes makes the odd pass observe the even
    // result, turning the intended Jacobi sweep into an order-dependent
    // Gauss-Seidel operator and substantially increasing diffusion.
    const diffusionPasses = [
      ["x", this.pipelines.diffuseGammaX],
      ["y", this.pipelines.diffuseGammaY],
      ["z", this.pipelines.diffuseGammaZ],
    ] as const;
    for (let iteration = 0; iteration < this.gammaDiffusionIterations; iteration += 1) {
      diffusionPasses.forEach(([axis, pipeline], index) => this.run(encoder,
        `Uniform gamma diffusion iteration ${iteration + 1}/${this.gammaDiffusionIterations} ${axis} Jacobi`,
        pipeline, this.gammaDiffusionGroups[index & 1]!));
      // Three axis passes leave their result in the A/B output pair. Restore
      // the density-advection ABI expected by sharpening and by the next
      // repetition.
      this.copyField(encoder,{ texture: this.volumeA }, { texture: this.volumeB },
        [this.info.nx, this.info.ny, this.info.nz]);
      this.copyField(encoder,{ texture: this.gammaB }, { texture: this.gammaA },
        [this.info.nx, this.info.ny, this.info.nz]);
    }
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.volumeB }, { texture: this.symmetryStageAuditFields.densityDiffusion },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.gammaA }, { texture: this.symmetryStageAuditFields.gammaPostDiffusion },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (this.gammaDiffusionIterations > 0) seam?.(UNIFORM_ADVANCE_PHASE.gammaDiffusion);
    if (this.densitySharpening) {
      encoder.clearBuffer(this.conditioningScratch);
      this.run(encoder, "Uniform interface sharpening", this.pipelines.sharpenCompute, this.sharpenComputeGroup);
      if (this.sharpeningMassCorrection) {
        seam?.(UNIFORM_ADVANCE_PHASE.interfaceSharpening);
        this.run(encoder, "Uniform conserved sharpening scatter", this.pipelines.sharpenScatter, this.sharpenScatterGroup);
        this.run(encoder, "Uniform conserved sharpening resolve", this.pipelines.sharpenResolve, this.sharpenResolveGroup);
        seam?.(UNIFORM_ADVANCE_PHASE.sharpeningMassCorrection);
      } else {
        // sharpenCompute writes volumeA while every downstream surface stage
        // reads volumeB. Preserve that ABI even for the deliberately
        // non-conservative one-pass ablation.
        this.copyField(encoder,
          { texture: this.volumeA }, { texture: this.volumeB },
          [this.info.nx, this.info.ny, this.info.nz],
        );
        seam?.(UNIFORM_ADVANCE_PHASE.interfaceSharpening);
      }
    }
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.volumeB }, { texture: this.symmetryStageAuditFields.densitySharpening },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (this.solidExcessCorrection && (activeBodies.length > 0 || sceneHasTerrain(this.scene))) {
      encoder.clearBuffer(this.conditioningScratch);
      this.run(encoder, "Uniform partial-solid excess scatter", this.pipelines.scatterSolidExcess, this.solidExcessScatterGroup);
      this.run(encoder, "Uniform partial-solid excess resolve", this.pipelines.resolveSolidExcess, this.solidExcessResolveGroup);
      seam?.(UNIFORM_ADVANCE_PHASE.solidExcess);
    }
    }
    this.copyField(encoder,{ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
    // Algorithm 1 steps 3-4: advect/force velocity after the surface-density
    // update, then enforce incompressibility.
    if (this.velocityTransport === "maccormack") {
      // CM11b Sec. 3.5 modified MacCormack with local-extrema fallback. Extend
      // the forward prediction before the reverse trace so every lookup has a
      // defined velocity; body forces are applied exactly once in correction.
      this.run(encoder, "Uniform bounded MacCormack velocity prediction",
        this.pipelines.advect, this.advectGroup);
      if (this.symmetryStageAuditFields) this.copyField(encoder,
        { texture: this.velocityC }, { texture: this.symmetryStageAuditFields.velocityPrediction },
        [this.info.nx, this.info.ny, this.info.nz],
      );
      this.encodeVelocityExtrapolation(encoder, true);
      this.run(encoder, "Uniform bounded MacCormack reverse advection",
        this.pipelines.reverse, this.reverseGroup);
      // velocityD is also the opt-in reverse-advection audit texture.
      this.run(encoder, "Uniform bounded MacCormack correction and body forces",
        this.pipelines.correct, this.correctGroup);
    } else {
      // The original one-pass path already performs the midpoint backward
      // trace and applies body forces exactly once to the advected field.
      this.run(encoder, "Uniform semi-Lagrangian velocity advection and body forces",
        this.pipelines.semiLagrangian, this.semiLagrangianGroup);
      if (this.symmetryStageAuditFields) this.copyField(encoder,
        { texture: this.velocityB }, { texture: this.symmetryStageAuditFields.velocityPrediction },
        [this.info.nx, this.info.ny, this.info.nz],
      );
      // The audit schema predates the selectable transport. Publish identity
      // placeholders for its MacCormack-only intermediate stages so a
      // semi-Lagrangian run never exposes stale texture contents as evidence.
      if (this.symmetryStageAuditFields) {
        this.copyField(encoder,
          { texture: this.velocityB }, { texture: this.velocityD },
          [this.info.nx, this.info.ny, this.info.nz],
        );
        this.copyField(encoder,
          { texture: this.transportA }, { texture: this.transportB },
          [this.info.nx + 2, this.info.ny + 2, this.info.nz + 2],
        );
      }
    }
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.velocityB }, { texture: this.symmetryStageAuditFields.velocityAdvection },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    seam?.(UNIFORM_ADVANCE_PHASE.advectionCorrection);
    this.encodeSurfaceDeficitBalance(encoder);
    this.pressureMultigrid.encode(encoder, this.pressureMultigridGroup,
      seam && ((stage) => seam(UNIFORM_PRESSURE_STAGE_PHASE[stage])),
      this.planPressureCycleBudget(), this.pressureRecoveryExpected);
    // The cycle counters are final the instant the solve is encoded, and this
    // copy adds no pass, so it cannot move a stage seam. It is encoded only
    // while the lagged budget is live and no earlier map is outstanding.
    let pressureCycleDemandEncoded = false;
    if (this.pressureCycleBudgetLagged && !this.pressureCycleDemandPending) {
      this.pressureCycleDemandReadback ??= this.device.createBuffer({
        label: "Uniform CM11a cycle demand readback", size: 32,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      encoder.copyBufferToBuffer(this.pressureMultigrid.diagnostics, 64,
        this.pressureCycleDemandReadback, 0, 28);
      pressureCycleDemandEncoded = true;
    }
    this.run(encoder, "Uniform pressure projection", this.pipelines.project, this.projectGroup);
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.velocityA }, { texture: this.symmetryStageAuditFields.pressureProjection },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    const combinedPublication = this.nativeRootExecution && this.geometricVolume;
    const coupledRigid = this.rigidCoupling && activeBodies.length > 0;
    if(!combinedPublication || coupledRigid)seam?.(UNIFORM_ADVANCE_PHASE.pressureProjection);
    if (this.rigidCoupling && activeBodies.length > 0) {
      this.run(encoder, "Uniform rigid-body coupling", this.pipelines.coupleRigid, this.rigidGroup);
      this.copyField(encoder,{ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
      this.copyField(encoder,{ texture: this.velocityB }, { texture: this.velocityA }, [this.info.nx, this.info.ny, this.info.nz]);
      encoder.copyBufferToBuffer(this.boundaryVelocityB, 0, this.boundaryVelocityA, 0, this.negativeBoundaryVelocityBytes);
      const cellVolume = c.width_m * c.height_m * c.depth_m / (this.info.nx * this.info.ny * this.info.nz);
      this.rigidSystem.encode(encoder, dt, cellVolume, 1, c.height_m / this.info.ny);
      if(!combinedPublication)seam?.(UNIFORM_ADVANCE_PHASE.rigidCoupling);
    }
    // Sec. 3.8 remains the optional global reconstruction. Container geometry
    // is not selected here; it is already present in SolidWorld.
    if (this.geometricVolume) {
      // This independent pass may finish before projection. Bypass the chain
      // boundary here and close their combined interval on diagnostics, which
      // consumes the projected fields. No artificial numerical dependency.
      this.run(combinedPublication ? rawEncoder : encoder, "Uniform Geometric surface publication", this.volumePipelines.uvPublish!, this.wallFilmResolveGroup);
    } else if (this.densityPostProcessing) {
      this.run(encoder, "Uniform post-process blur x", this.pipelines.postprocessBlurX, this.postprocessBlurXGroup);
      this.run(encoder, "Uniform post-process blur y", this.pipelines.postprocessBlurY, this.postprocessBlurYGroup);
      this.run(encoder, "Uniform post-process blur z", this.pipelines.postprocessBlurZ, this.postprocessBlurZGroup);
      this.run(encoder, "Uniform sub-grid surface resolve", this.pipelines.postprocessResolve, this.postprocessResolveGroup);
    } else {
      this.run(encoder, "Uniform wall-film resolve", this.pipelines.wallFilmResolve, this.wallFilmResolveGroup);
    }
    this.fieldPages?.encodePublications(encoder);
    seam?.(combinedPublication
      ? coupledRigid ? {...UNIFORM_ADVANCE_PHASE.rigidCoupling,label:"Rigid coupling + surface publication"}
        : {...UNIFORM_ADVANCE_PHASE.pressureProjection,label:"Pressure projection + surface publication"}
      : this.geometricVolume ? UNIFORM_VOLUME_PHASE.surface : UNIFORM_ADVANCE_PHASE.densityPostProcess);
    // The final phase closes on the reduction pass itself (its end-of-pass
    // counter) rather than on a synthetic marker pass after it: a marker
    // touches no frame resource, so Metal is free to schedule it early and its
    // timestamp lands before the boundary it is meant to close.
    // The box is final the moment the active region is finalized, and these
    // copies add no compute pass, so they cannot move a stage seam.
    let windowReadbackEncoded = false;
    if (this.activeRegionEnabled && !this.windowDispatchIndirect && !this.windowReadbackPending) {
      this.windowReadback ??= this.device.createBuffer({
        label: "Uniform solve-window box readback", size: 80,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      encoder.copyBufferToBuffer(this.activeRegion, 0, this.windowReadback, 0, 64);
      encoder.copyBufferToBuffer(this.activeRegion, UNIFORM_ACTIVE_VIOLATION_WORD * 4,
        this.windowReadback, 64, 8);
      encoder.copyBufferToBuffer(this.activeRegion, UNIFORM_ACTIVE_TRAVEL_TOTAL_WORD * 4,
        this.windowReadback, 72, 4);
      this.windowReadbackStep = this.windowStep;
      windowReadbackEncoded = true;
    }
    physicsTrace?.completeFinalPhaseOnNextPass(UNIFORM_ADVANCE_PHASE.diagnosticsReduction);
    this.diagnosticsReductionOwed = uniformAbOn("lazystats") && !shouldTracePhysics;
    if (!this.diagnosticsReductionOwed) this.run(encoder, "Uniform diagnostics reduction", this.pipelines.reduce, this.reductionGroup);
    physicsCPUTrace?.completePhase(UNIFORM_ADVANCE_PHASE.diagnosticsReduction);
    physicsTrace?.resolve(encoder);
    physicsQueueTrace?.begin();
    this.device.queue.submit([encoder.finish()]);
    if (pressureCycleDemandEncoded) this.readPressureCycleDemand();
    if (windowReadbackEncoded) this.readWindowBox();
    if (physicsCPUTrace) {
      this.info.physicsCPUTrace = physicsCPUTrace.finish({ id: "other", label: "Capture closure + command submission" });
      this.info.physicsCaptureIdentity = {
        sampleId: physicsTraceSampleId,
        context: physicsTraceContext,
        frameId: gpuPhysicsPerformanceActivityFrameId({
          sampleId: physicsTraceSampleId, context: physicsTraceContext,
        }),
      };
    }
    // Prefer the hardware partition; one unusable sample retires it for this
    // solver and the queue-wall observation carries the panel from then on.
    const physicsQueueTraceRead = physicsQueueTrace?.read(this.device.queue);
    const hardwarePhysicsTraceRead = physicsTrace?.read();
    const physicsTraceRead = hardwarePhysicsTraceRead
      ? hardwarePhysicsTraceRead
        .then((trace) => { this.hardwarePhysicsTraceInvalid = !trace; return trace ?? physicsQueueTraceRead; })
        .catch(() => { this.hardwarePhysicsTraceInvalid = true; return physicsQueueTraceRead; })
      : physicsQueueTraceRead;
    if (physicsTraceRead) {
      this.lastPhysicsTraceAt_ms = traceRequestedAt_ms;
      this.physicsTracePending = true;
      void physicsTraceRead.then((trace) => {
        const current = usePerformanceInstrumentationStore.getState();
        if (trace && !this.disposed && current.enabled && current.enabledAt_ms <= traceRequestedAt_ms) {
          this.info.physicsTrace = trace;
        }
      }).catch(() => {}).finally(() => { this.physicsTracePending = false; });
    }
    this.info.submittedTime_s = this.lastTime;
    this.info.simulatedTime_s = this.lastTime;
    const submittedTime = this.lastTime;
    void this.device.queue.onSubmittedWorkDone().then(() => {
      if (!this.disposed) this.info.completedTime_s = Math.max(this.info.completedTime_s ?? 0, submittedTime);
    }).catch(() => {});
    return true;
  }

  async readStats(): Promise<GPUEulerianInfo> {
    await this.awaitFrameCompletion();
    if (this.disposed || this.readbackPending) return this.info;
    this.readbackPending = true;
    this.statsReadback ??= this.device.createBuffer({ label: "Uniform reference diagnostics readback", size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Uniform reference diagnostics readback" });
    this.encodeOwedDiagnosticsReduction(encoder);
    encoder.copyBufferToBuffer(this.reductions, 0, this.statsReadback, 0, 32);
    encoder.copyBufferToBuffer(this.reductions, 32, this.statsReadback, 248, 8);
    if (this.volumePageConfig) encoder.copyBufferToBuffer(this.conditioningScratch, 4 * (this.volumePageConfig.base + 4), this.statsReadback, 236, 4);
    if(this.volumeWorkCounts)encoder.copyBufferToBuffer(this.volumeWorkCounts,0,this.statsReadback,240,8);
    // Only the step that ran the classify dispatch leaves a meaningful count.
    const tileMap = this.sharpenTileMapEncoded;
    if (tileMap) encoder.copyBufferToBuffer(this.conditioningScratch, this.sharpenTileCountWordOffset, this.statsReadback, 192, 4);
    if (this.twoLevelEncoded) encoder.copyBufferToBuffer(this.conditioningScratch,
      this.twoLevelShellCountOffset, this.statsReadback, 196,
      UNIFORM_VOLUME_TWO_LEVEL_COUNTER_WORDS * 4);
    encoder.copyBufferToBuffer(this.pressureMultigrid.diagnostics, 0, this.statsReadback, 32, 60);
    encoder.copyBufferToBuffer(this.pressureMultigrid.diagnostics, 64, this.statsReadback, 176, 12);
    encoder.copyBufferToBuffer(this.pressureMultigrid.diagnostics, 76, this.statsReadback, 208, 28);
    encoder.copyBufferToBuffer(this.velocityExtrapolator.convergenceDiagnostics, 0, this.statsReadback, 96, 16);
    encoder.copyBufferToBuffer(this.activeRegion, 0, this.statsReadback, 112, 64);
    this.device.queue.submit([encoder.finish()]);
    try {
      await this.statsReadback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(this.statsReadback.getMappedRange().slice(0));
      // The same counters the lagged budget reads on its own buffer. A caller
      // that polls stats every step (the Dawn probes and harness do) therefore
      // keeps the demand signal fresh even when the solver's own copy was
      // skipped because an earlier map was still outstanding.
      if ((this.info.encodedSteps ?? 0) > 0) {
        this.pressureCyclesExecutedSample = words[45]! + words[46]!;
        this.pressureCycleConvergedSample = words[44] === 1;
      }
      if (this.volumePageConfig) this.info.uniformVolumePagesActive = words[59]!;
      if(this.volumeWorkCounts){
        this.info.uniformVolumeTransportWorkgroups=words[60]!;
        this.info.uniformVolumeSharpenWorkgroups=this.densitySharpening?words[61]!:0;
      }
      if(this.pageDomain){
        this.info.uniformPageMissingReadFields=words[62]!;
        this.info.uniformPageMissingReads=words[63]!;
      }
      const reference = Math.max(1, this.referenceVolumeCells);
      this.info.representedVolumeCellSum = words[0] / 2048;
      this.info.volumeCellSum = words[3] / 2048;
      this.info.representedVolumeDrift = (this.info.representedVolumeCellSum - reference) / reference;
      this.info.rawVolumeDrift = (this.info.volumeCellSum - reference) / reference;
      this.info.volumeDrift = this.info.rawVolumeDrift;
      this.info.volumeTelemetrySource = "dense-volume";
      this.info.front_m = -this.scene.container.width_m / 2
        + words[1] * this.scene.container.width_m / this.info.nx;
      this.info.frontTelemetrySource = "dense-volume";
      this.info.maxSpeed_m_s = new Float32Array(new Uint32Array([words[2]]).buffer)[0];
      Object.assign(this.info, { uniformUnplaceableSolidExcess_cells: words[4] / 2048 });
      Object.assign(this.info, {
        uniformCM11aResidualInfinity: new Float32Array(new Uint32Array([words[8]]).buffer)[0],
        uniformCM11aConverged: words[11] === 0,
        uniformCM11aCoarseIterations: words[10],
        uniformCM11aCycleConverged: words[44] === 1,
        uniformPressureAcceptedResidual: new Float32Array(new Uint32Array([words[52]]).buffer)[0],
        uniformPressureRejectedCycles: words[53],
        uniformPressureRecoverySweeps: words[54],
        uniformPressureInitialResidual: new Float32Array(new Uint32Array([words[57]]).buffer)[0],
        uniformPressureRecoveryExhausted: words[58] !== 0,
        uniformCM11aFullCyclesExecuted: words[45],
        uniformCM11aVCyclesExecuted: words[46],
        uniformPressureCyclesExecuted: words[45]! + words[46]!,
        uniformPressureCyclesConverged: words[44] === 1,
        uniformCM11aCapFailure: words[11] !== 0,
        uniformCM11aFailingCoarseInvocation: words[12],
        uniformCM11aCoarseMaxAbsRhs: new Float32Array(new Uint32Array([words[13]]).buffer)[0],
        uniformCM11aCoarseMaxDiagonalPressure: new Float32Array(new Uint32Array([words[14]]).buffer)[0],
        uniformCM11aCoarseMaxAbsPressure: new Float32Array(new Uint32Array([words[15]]).buffer)[0],
        uniformCM11aCoarseProjectedGapPressure: new Float32Array(new Uint32Array([words[16]]).buffer)[0],
        uniformCM11aCoarseNormalizedProjectedResidual: new Float32Array(new Uint32Array([words[17]]).buffer)[0],
        uniformCM11aFineResidualInfinity: new Float32Array(new Uint32Array([words[18]]).buffer)[0],
        uniformCM11aFineProjectedGapPressure: new Float32Array(new Uint32Array([words[19]]).buffer)[0],
        uniformCM11aCoarseActiveRows: words[20],
        uniformCM11aCoarseFreeRows: words[21],
        uniformCM11aCoarseWorstRow: words[22] & 0x3fff_ffff,
        uniformCM11aCoarseWorstRowActive: (words[22] & 0x4000_0000) !== 0,
        uniformCM11aCoarseWorstRowHalo: (words[22] & 0x8000_0000) !== 0,
        uniformFIMTerminalActiveFaces: words[24] + words[25],
        uniformFIMConverged: words[24] + words[25] === 0,
        uniformFIMExecutedPasses: words[27],
      });
      const activeMinimum = { x: words[35]!, y: words[36]!, z: words[37]! };
      const activeMaximum = { x: words[38]!, y: words[39]!, z: words[40]! };
      const activeCellCount = Math.max(0, activeMaximum.x - activeMinimum.x)
        * Math.max(0, activeMaximum.y - activeMinimum.y)
        * Math.max(0, activeMaximum.z - activeMinimum.z);
      Object.assign(this.info, {
        uniformActiveRegionMinimum: activeMinimum,
        uniformActiveRegionMaximum: activeMaximum,
        uniformActiveRegionCellCount: activeCellCount,
        uniformActiveRegionFraction: activeCellCount / Math.max(1, this.info.cellCount),
        uniformSolveWindowDispatch: this.activeRegionEnabled
          ? (this.windowDispatchIndirect ? "indirect" : "host") : undefined,
        uniformSolveWindowClippedSteps: this.activeRegionEnabled && !this.windowDispatchIndirect
          ? this.windowViolations : undefined,
        uniformSolveWindowDenseSteps: this.activeRegionEnabled && !this.windowDispatchIndirect
          ? this.windowDenseSteps : undefined,
        uniformPressureLattice: this.pressureWindowLattice
          ? `${this.pressureWindowCapacity.join("x")} @ ${this.pressureWindowOrigin.join(",")}`
          : undefined,
        uniformPressureLatticeWindowed: this.pressureWindowLattice
          ? this.pressureWindowLatticeActive : undefined,
        uniformPressureLatticeReplans: this.pressureWindowLattice
          ? this.pressureWindowReplans : undefined,
        uniformPressureLatticeReplanMs: this.pressureWindowLattice
          ? this.pressureWindowReplanMs : undefined,
        uniformSolveWindowMaxLagSteps: this.activeRegionEnabled && !this.windowDispatchIndirect
          ? this.windowMaxLagSteps : undefined,
      });
      this.info.uniformSharpenWorkMap = tileMap;
      this.info.uniformSharpenTilesActive = tileMap ? words[48]! : undefined;
      this.info.uniformSharpenTilesTotal = tileMap ? this.sharpenTileCount : undefined;
      // The floor's own price: cells zeroed across the gather and the eight
      // commit sweeps, and the mass that went with them in sixty-fourths of
      // the threshold. Off, both are structurally zero.
      const dust = this.geometricVolume && this.volumeDustThreshold > 0;
      Object.assign(this.info, {
        uniformVolumeDustThreshold: this.geometricVolume ? this.volumeDustThreshold : undefined,
        uniformVolumeDustCells: dust ? words[5]! : undefined,
        uniformVolumeDustMass_cells: dust ? words[6]! * this.volumeDustThreshold / 64 : undefined,
        uniformTwoLevelVelocity: this.geometricVolume ? this.twoLevelEncoded : undefined,
        uniformTwoLevelFineReach: this.geometricVolume ? this.twoLevelFineReach : undefined,
        uniformTwoLevelFineTiles: this.twoLevelEncoded ? words[7]! : undefined,
        uniformTwoLevelTilesTotal: this.twoLevelEncoded ? this.twoLevelTileCount : undefined,
        uniformTwoLevelShellTiles: this.twoLevelEncoded ? words[49]! : undefined,
        uniformTwoLevelShellReach: this.geometricVolume ? this.twoLevelShellReach : undefined,
        uniformTwoLevelExtensionTiles: this.twoLevelEncoded ? this.twoLevelExtensionTiles : undefined,
        uniformTwoLevelAdvectionTiles: this.twoLevelEncoded ? this.twoLevelAdvectionTiles : undefined,
      });
      // E3. The measured displacement is the domain maximum of |v|*dt/h in
      // cells, taken at the head of the step on the velocity transport traces.
      // The required reach is that step's ceil(D)+1 cells in whole tiles, which
      // is what the dilation used plus the margin -- so "used below required"
      // can only mean the shader's cap bit, and the panel says so.
      const displacement = this.twoLevelEncoded
        ? new Float32Array(new Uint32Array([words[51]!]).buffer)[0]! : 0;
      const requiredReach = Math.ceil((Math.ceil(Math.max(displacement, 0)) + 1) / 4);
      Object.assign(this.info, {
        uniformTransportWorkMap: this.geometricVolume ? this.transportTilesEncoded : undefined,
        uniformTransportTiles: this.transportTilesEncoded ? words[50]! : undefined,
        uniformTransportTilesTotal: this.twoLevelEncoded ? this.twoLevelTileCount : undefined,
        uniformTransportReachMargin: this.geometricVolume ? this.transportReach : undefined,
        uniformTransportReachTiles: this.twoLevelEncoded
          ? Math.max(0, Math.min(16, requiredReach + this.transportReach)) : undefined,
        uniformTransportMaxDisplacement_cells: this.twoLevelEncoded ? displacement : undefined,
        uniformTransportRequiredReachTiles: this.twoLevelEncoded ? requiredReach : undefined,
      });
      return this.info;
    } finally {
      if (this.statsReadback.mapState === "mapped") this.statsReadback.unmap();
      this.readbackPending = false;
    }
  }

  enableCM11aCoarsestCapture(invocation = 1): void {
    this.pressureCoarsestCapture = invocation;
    this.pressureMultigrid.enableCoarsestCapture(invocation);
  }
  readCM11aCoarsestCapture(): Promise<UniformCM11aCoarsestCapture | undefined> {
    return this.pressureMultigrid.readCoarsestCapture();
  }

  /** A voxel stroke reaches this solver as solid-mask bits, so only a different lattice is refused. */
  validateLiveSolidEdit(scene: SceneDescription): void {
    const [nx, ny] = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
    if (nx !== this.info.nx || ny !== this.info.ny) throw new Error("This voxel edit changes the fluid lattice; reset the scene to apply it.");
  }

  applySceneUniforms(scene: SceneDescription): void {
    this.scene = scene;
    this.faceAuthorityStored = false;
    const dirty = this.solidMask.update(solidWorldForScene(scene));
    if (dirty) this.device.queue.writeBuffer(this.activeScratch,
      (this.solidVoxelScratchOffsetWords + dirty.firstWord) * 4, this.solidMask.words.buffer as ArrayBuffer,
      dirty.firstWord * 4, dirty.wordCount * 4);
    this.inflowBoundary = scene.fluid.inflow
      ? createInflowGridBoundary(scene.fluid.inflow, scene.container, [this.info.nx, this.info.ny, this.info.nz])
      : undefined;
    this.pressureMultigrid.setDepthSymmetry(scene.container.depthBoundary === "symmetry");
    this.activeRegionRescanPending = true;
  }

  get rigidRenderBuffer(): GPUBuffer { return this.rigidSystem.renderBuffer; }
  get rigidMotionBuffer(): GPUBuffer { return this.rigidSystem.motionBuffer; }
  setSelectedRigidBody(index: number): void { this.rigidSystem.setSelectedIndex(index); }
  pickRigidBody(origin: RigidBodyState["position_m"], direction: RigidBodyState["position_m"]) { return this.rigidSystem.pick(origin, direction); }
  readRigidBodyPoses() { return this.rigidSystem.readPoses(); }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const texture of new Set([
      this.velocityA, this.velocityB, this.velocityC, this.velocityD,
      this.pressureA, this.pressureB, this.volumeA, this.volumeB,
      this.surfaceA, this.surfaceB, this.gammaA, this.gammaB,
      ...Object.values(this.symmetryStageAuditFields ?? {}),
      this.heightA, this.heightB, this.terrainTexture,
      this.transportA, this.transportB,
    ])) texture.destroy();
    this.surfaceVolumeCorrection?.destroy();
    this.vertexPhiField?.destroy(); this.vertexPhiScratch?.destroy();
    if(this.scratchArena)this.scratchArena.destroy();
    else { this.volumeEdges?.destroy(); this.volumeDonorSums?.destroy(); }
    this.boundaryVelocityA.destroy(); this.boundaryVelocityB.destroy();
    this.boundaryVelocityC.destroy(); this.boundaryVelocityD.destroy();
    this.symmetryStageAuditNegativeBoundaryVelocity?.destroy();
    this.symmetryStageAuditBetaBuffer?.destroy();
    this.macCormackAuditBinding.destroy();
    this.velocityExtrapolator.destroy();
    for (const instance of this.pressureInstances.values()) instance.destroy();
    this.pressureInstances.clear();
    this.discardPressurePrewarm();
    this.params.destroy();
    this.reductions.destroy();
    this.conditioningScratch.destroy();
    this.activeRegion.destroy();
    this.activeScratch.destroy();
    this.activeDispatch.destroy();
    this.phiRegion?.destroy();this.phiDispatch?.destroy();
    this.fieldPages?.destroy();
    this.pageDomainPublication?.destroy();
    this.volumeWorkDispatch?.destroy();
    this.volumeWorkCounts?.destroy();
    this.volumePageSharpenFlag?.destroy();
    this.volumeTransportPageView?.destroy();
    this.rigidSystem.destroy();
    this.rigidExchange.destroy();
    this.statsReadback?.destroy();
    // An outstanding map rejects on destroy; `readPressureCycleDemand` catches
    // it and the `disposed` guard keeps it from touching a dead solver.
    this.pressureCycleDemandReadback?.destroy();
  }
}
