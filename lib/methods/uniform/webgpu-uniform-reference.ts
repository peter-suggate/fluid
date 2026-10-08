import {UniformMixedDiagnostics} from "./uniform-mixed-diagnostics";
import { UNIFORM_MIXED_RECEIPT_RING, UniformMixedFrame, type UniformMixedFrameReceipt, type UniformMixedFrameRelayout, type UniformMixedFrameTrace } from "./uniform-mixed-frame";
import { uniformMixedFixedFineCapacity } from "./uniform-mixed-ownership";
import { uniformDetailCapacityRule } from "./uniform-detail-policy";
import { uniformMixedPressureRootBytes } from "./uniform-mixed-pressure-memory";
import { assertUniformMixedOptions } from "./uniform-mixed-options";
import { assertUniformMixedSolidPromotion, createUniformMixedLayout, createUniformMixedLayoutFromWidths, uniformMixedLiquidSolidPromotion, uniformMixedSolidTiles } from "./uniform-mixed-layout";
import { sameUniformDetailSettings, UNIFORM_DETAIL_PARAM_KEYS, uniformDetailCoarseSolids, uniformDetailCoarseSurface, uniformDetailHeldDistance, uniformDetailImportance, uniformDetailRequests, uniformDetailSettings, uniformDetailValues, type UniformDetailSettings } from "./uniform-detail-policy";
import { buildUniformDetailRequests, uniformDetailDomain, uniformDetailRequestKey, uniformRegionDetailTier, type DetailRequest } from "./uniform-detail-requests";
import { UniformDetailPlanner, UniformDetailRequestTiles, uniformDetailTileBounds, type UniformDetailPlan } from "./uniform-detail-planner";
import type { UniformWorkEdit } from "./uniform-buffered-work";
import type { SolverDetailFocus, SolverDetailInput } from "../../core/solver-detail";
import { UniformMixedLayoutBuilder, UNIFORM_MIXED_RELAYOUT_RECEIPT } from "./uniform-mixed-layout-builder";
import { UniformMixedDynamicClassifier, UNIFORM_MIXED_DYNAMIC_FULL_TOLERANCE } from "./uniform-mixed-dynamic";
import { refinementRegionLattice } from "../../core/refinement-regions";
import { UniformScratchArena } from "./uniform-scratch-arena";
import { uniformBrickCells } from "./uniform-volume-donor-sum.wgsl";
import { uniformPageHasRectangularCoverage } from "./uniform-page-execution";
import { UniformTexturePages } from "./uniform-texture-pages";
import {initialUniformPageDomain,type UniformPageDomain} from "./uniform-page-domain";
import { uniformVolumeWorkLayout, type UniformVolumePageShaderOptions } from "./uniform-volume-pages.wgsl";
import { uniformDensityPostProcessingEnabled } from "./uniform-options";
export { uniformDensityPostProcessingEnabled } from "./uniform-options";
import type { DenseLevelSetVolumeConsumerSource } from "../../core/levelset-consumer-abi";
import {
  SOLVE_WINDOW_HOST_GROUPS_WORD, SOLVE_WINDOW_RECORD_WORDS,
  type GPUFluidSolveWindowSource,
} from "../../core/method-view-records";
import {
  UNIFORM_VOLUME_EDGE_BYTES,
  UNIFORM_VOLUME_TWO_LEVEL_COUNTER_WORDS,
  UNIFORM_VOLUME_TWO_LEVEL_WORDS_PER_TILE,
} from "./uniform-volume.wgsl";
import { createUniformReferenceComputeShader } from "./webgpu-uniform-reference.wgsl";
import { uniformAbOn } from "./uniform-ab-switch";
import { uniformVolumeInitialPhi, uniformInitialVolume } from "./uniform-volume-initial";
import { averageInflowStrength, createInflowGridBoundary, type InflowGridBoundary } from "../../core/inflow-boundary";
import type { SceneDescription } from "../../core/model";
import { planUniformHostAllocation } from "./uniform-host-allocation";
import { boundingRadius, initializeRigidBodies, type RigidBodyState } from "../../core/rigid-body";
import { UniformMixedBodies } from "./uniform-mixed-bodies";
import { UniformPipelineNeeds, type UniformPipelineNeed } from "./uniform-pipeline-needs";
import { UniformPrescribedSolidMotion } from "./uniform-prescribed-solid-motion";
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
  UNIFORM_PRESSURE_RESIDUAL_TOLERANCE,
  UNIFORM_CM11A_V_CYCLES,
  uniformCM11aCycleBudget,
  WebGPUUniformPressureMultigrid,
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
import { uniformGeometricSharpeningSweeps, uniformGeometricSurfaceVolumeRounds } from "./uniform-geometric-parameters";
export { UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS, UNIFORM_GAMMA_DIFFUSION_MAX_ITERATIONS } from "./parameters";
import { uniformFixedAdvanceReady, uniformFixedStep_s } from "./uniform-paper";
import { liveFluidEditRefusal, type LiveFluidEdit, type LiveFluidEditResult } from "../../core/live-fluid-edit";
import { SOLID_OCCUPANCY_MASK_HEADER_WORDS, SolidOccupancyMask } from "../../core/solid-occupancy-mask";
import { solidWorldForScene } from "../../core/solid-world";
import { UniformDetailStorage, uniformDetailDenseShader, uniformDetailField, uniformDetailStorageOptions, type UniformDetailClass, type UniformDetailOptions } from "./uniform-detail-fields";


export { UNIFORM_PAPER_DT_S } from "./uniform-paper";

export interface WebGPUUniformReferenceOptions {
  /** Experimental particle velocity transport in the surface band. */
  narrowBandFlip?: boolean;
  narrowBandCoarseParticles?: boolean;
  /** Independent dense vertex level set and conservative cell volume, advanced
   * only by the mixed-ownership frame. */
  geometricVolume?: boolean;
  /** Full-lattice pressure smoothing control for the work-list regression. */
  pressureSmoothingForQA?: "dense";
  /** Recompute finest RHS interface/capacity instead of consuming built topology. */
  pressureAuthorityForQA?: "raw";
  /** Original coefficient bake and single-workgroup surface balance. */
  systemBuildForQA?: "baseline";
  /** Dense extension launches and repeated neighbor convergence queries. */
  extensionWorkForQA?: "baseline";
  /** QA-only former pressure layouts; production uses native execution fields. */
  pressureStorageForQA?: "paged" | "paged-logical";
  geometricRedistance?: boolean;
  /** Retain stage fields, including inactive cells, for full-field diagnostics. */
  retainStageDiagnosticsForQA?: boolean;
  /** QA: h-field detail storage options (uniform-detail-fields.ts), over the
   * process QA override and the production default. */
  detailStorageForQA?: UniformDetailOptions;
  /** Global surface-volume constraint; defaults on for 3D geometric volume. */
  totalSurfaceVolume?: boolean;
  surfaceDeficitBalancing?: boolean;
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
  /** Extra floor for dilute orphan V only; zero disables it. */
  orphanDustThreshold?: number;
  /** Simulation detail controls (uniform-detail-policy.ts). Interim mapping
   * onto this solver's ownership: Requested and Full build the CPU region
   * layout from the detail planner's admitted h tiles on a 4h background;
   * Dynamic runs the GPU surface census (docs/plans/uniform-dynamic-coarsening.md)
   * with the planner's h tiles as statics. */
  detail?: UniformDetailSettings;
  mixedCoarseningReach?: number;
  mixedCoarseningHysteresis?: number;
  /**
   * Geometric only: the splash-survival controls
   * (docs/uniform-geometric-splash-dissipation-plan.md), both on by default.
   *  - phiCubicAdvection: clamped Catmull-Rom instead of trilinear for the
   *    advected phi of band vertices.
   *  - phiDrain: raise phi-liquid vertices with no V anywhere around them.
   *  - phiPreserveSurface (off by default): redistancing keeps h vertices
   *    beside the surface at their advected value (CM11b sec. 3.4).
   */
  phiCubicAdvection?: boolean;
  phiDrain?: boolean;
  phiPreserveSurface?: boolean;
  /** GPU-resident sparse work boxes; false retains the original dense control. */
  activeRegion?: boolean;
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
  /** Geometric only: mixed sharpening sweeps per step (even). */
  sharpeningSweeps?: number;
  /** Geometric only: secant Newton rounds of the total surface-volume shift. */
  surfaceVolumeRounds?: number;
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
  timeStep?: "paper" | "sixtieth" | "scene";
  deferPipelineCompilation?: boolean;
}

// Sec. 3.4 permits one through seven gamma-diffusion repetitions per time
// step. One repetition is the reference schedule: each dimensional sweep is
// one Jacobi update from a single snapshot, as specified by LAF11/CM12. More
// repetitions remain available explicitly, but are not a neutral robustness
// setting: they apply more physical/numerical diffusion per simulation step.


/** Pipelines built per round of the background warm-up. Metal compiles
 * serially (1, 4 and all at once take the same 44-45 s on dam64 and none
 * moves the frame), so one a round: a change that waits queues behind one. */
const UNIFORM_PIPELINE_WARM_PACE = 1;
/**
 * No cell of this lattice is covered by a static solid voxel.
 *
 * Only the lattice's own cells count. The mask also carries a one-cell halo,
 * and every box scene's container is compiled into that halo as a one-voxel
 * shell (solidVoxelShellForScene), so an OR over all its words refused the
 * certificate to every tank. No consumer can see the halo: cellOpenFraction
 * returns zero for any cell outside the lattice before it reads the mask, and
 * each E4 site either clamps into the lattice or skips invalid cells. The test
 * is one masked run of nx bits per interior (y, z) row. It is recomputed only
 * when `update` reports a dirty range -- a voxel stroke or a scene load --
 * never per step.
 */
function uniformSolidMaskEmpty(mask: SolidOccupancyMask): boolean {
  const words = mask.words;
  if (!uniformAbOn("solidinterior")) {
    for (let index = SOLID_OCCUPANCY_MASK_HEADER_WORDS; index < words.length; index += 1) {
      if (words[index] !== 0) return false;
    }
    return true;
  }
  const sx = words[1]!, sy = words[2]!, sz = words[3]!;
  for (let z = 1; z < sz - 1; z += 1) for (let y = 1; y < sy - 1; y += 1) {
    const first = sx * (y + sy * z) + 1, last = first + sx - 3;
    for (let word = first >>> 5; word <= last >>> 5; word += 1) {
      const low = Math.max(first, word << 5) & 31, high = Math.min(last, (word << 5) + 31) & 31;
      const bits = (0xffffffff >>> (31 - high + low)) << low;
      if ((words[SOLID_OCCUPANCY_MASK_HEADER_WORDS + word]! & bits) !== 0) return false;
    }
  }
  return true;
}

interface UniformReferencePipelines {
  planPhiCensus: GPUComputePipeline;
  scanPhiSupport: GPUComputePipeline;
  scanActiveRegion: GPUComputePipeline;
  scanExternalActiveSources: GPUComputePipeline;
  reduceActiveRegionSummaries: GPUComputePipeline;
  reduceExternalActiveRegionSummaries: GPUComputePipeline;
  reducePhiSupportSummaries: GPUComputePipeline;
  finalizeActiveRegion: GPUComputePipeline;
  finalizePhiRegion: GPUComputePipeline;
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

/** The mixed frame's 4h continuation solves the mixed operator's surface rows. */
function mixedSurfaceThetaFragment(fragment: string): string {
  const marker = "const MG_MIXED_SURFACE_THETA:bool=false;";
  if (!fragment.includes(marker)) throw new Error("CM11a fragment lost its mixed surface-theta switch");
  return fragment.replace(marker, "const MG_MIXED_SURFACE_THETA:bool=true;");
}

/** CM11a entries the mixed 4h continuation's setup dispatches (the mixed
 * pressure root runs the traversal itself). */
const UNIFORM_MIXED_CONTINUATION_ENTRIES = ["mgDownsampleTopology", "mgExtrapolatePhiOneCell", "mgBakeCoefficients",
  "mgBuildSmoothTiles"] as const;

const PIPELINES = [
  ["planPhiCensus", "Plan phi support census", "planPhiCensus", false],
  ["scanPhiSupport", "Scan phi support", "scanPhiSupport", false],
  ["scanActiveRegion", "Scan active liquid bounds", "scanActiveRegion", false],
  ["scanExternalActiveSources", "Scan external active sources", "scanExternalActiveSources", false],
  ["reduceActiveRegionSummaries", "Reduce active liquid summaries", "reduceActiveRegionSummaries", false],
  ["reduceExternalActiveRegionSummaries", "Reduce external-source summaries", "reduceExternalActiveRegionSummaries", false],
  ["reducePhiSupportSummaries", "Reduce tile-padded phi support summaries", "reducePhiSupportSummaries", false],
  ["finalizeActiveRegion", "Finalize active liquid dispatches", "finalizeActiveRegion", false],
  ["finalizePhiRegion", "Finalize tile-padded phi region", "finalizePhiRegion", false],
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
/** The native kernels' h fields by binding name (uniformDetailDenseShader):
 * what the t=0 authority and publication groups bind there. The pressure
 * and transport slots are 1^3 placeholders under Geometric. */
const UNIFORM_REFERENCE_DETAIL_CLASSES: Readonly<Record<string, UniformDetailClass>> = {
  velocityIn: "face", velocityOut: "face", predictedVelocityIn: "face", reversedVelocityIn: "face",
  volumeIn: "cell", volumeOut: "cell", velocityPhaseIn: "cell", surfaceIn: "cell", gammaIn: "cell", gammaOut: "cell",
  uvPhiIn: "vertex", uvPhiOut: "vertex",
};
const UNIFORM_PHYSICS_TRACE_CADENCE_MS = 100;
/** Page edge of the geometric catalogue and volume page config. */
const UNIFORM_GEOMETRIC_PAGE_EDGE = 32;
/** Params lanes the mixed frame's shader still reads as constants: the fine
 * reach and transport margin in 4h tiles, and the phi agreement clamp. */
const UNIFORM_TWO_LEVEL_FINE_REACH = 2;
const UNIFORM_TRANSPORT_REACH = 1;
const UNIFORM_PHI_AGREEMENT_CLAMP = 0.02;

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
/** Relayout receipts (UNIFORM_MIXED_RELAYOUT_RECEIPT) in flight to the host
 * for diagnostics; a frame that finds none free reports nothing. */
const UNIFORM_MIXED_RELAYOUT_READS = 3;

export class WebGPUUniformReferenceSolver implements GPUSolverInstance {
  private readonly executionInfo: GPUEulerianInfo;
  get info(): GPUEulerianInfo { this.refreshMixedAllocation();return this.executionInfo; }
  get simulationCellScale(): 1 { return 1; }
  private readonly narrowBandFlip: boolean;
  private readonly narrowBandCoarseParticles: boolean;
  get particleSource(){return this.mixedFrame?.narrowBandFlip?.particleSource;}
  get narrowBandFlipInfo(){const stage=this.mixedFrame?.narrowBandFlip;return stage?{particles:stage.count,capacity:stage.capacity,reseedClipped:stage.reseedClipped,bandWidth:5,flipRatio:0.95,...stage.diagnostics}:undefined;}
  private readonly geometricVolume: boolean;
  private geometricRedistance: boolean;
  /** Sec. 3.4/3.5 rounding-residue floor in cell volumes; 0 is off. */
  private volumeDustThreshold: number;
  private orphanDustThreshold: number;
  private appliedRuntimeValues?: MethodParamValues;
  /** Two-level velocity sampling; off when the lattice has no 4h table. */
  private twoLevelVelocity: boolean;
  /** How far past the fine set the extension must still be exact, in 4h tiles. */
  private readonly twoLevelShellReach: number;
  /** Derived from detail.policy: "dynamic" is the census with its importance criteria, "regions" requests only. */
  private mixedCoarsening: "regions" | "dynamic";
  private detail: UniformDetailSettings;
  private detailPlanner?: UniformDetailPlanner;
  /** Latest transient focus, and the one the accepted plan used. */
  private detailFocusPending?: SolverDetailFocus;
  private detailFocus?: SolverDetailFocus;
  /** Accepted mixed frames: the planner's step clock (hysteresis, expiry). */
  private detailAcceptedStep = 0;
  /** The last committed plan, its Coarse-suppressed tiles and unconstrained-region count. */
  private detailPlan?: UniformDetailPlan;
  private detailSuppressed?: Uint8Array;
  private detailUnconstrained = 0;
  private mixedCoarseningReach: number;
  private mixedCoarseningHysteresis: number;
  /** Split pressure's own surface target and centre phi (UniformMixedFrameFields.pressureGeometry). */
  private mixedPressureGeometry?: { target: GPUTexture; centerPhi: GPUTexture };
  private totalSurfaceVolume: boolean;
  private surfaceDeficitBalancing: boolean;
  private readonly surfaceDeficitBalanceBytes: number;
  /** Splash-survival controls; see the option docs. */
  private phiCubicAdvection: boolean;
  private phiDrain: boolean;
  private phiPreserveSurface: boolean;
  /** Coarse cells whose E1 tables fit the conditioning plane; 0 disables E1. */
  private twoLevelTileCount: number;
  private readonly vertexPhiField?: GPUTexture;
  get vertexPhiTexture(): GPUTexture | undefined {
    if (!this.vertexPhiField) return undefined;
    // A base the detail storage adopted (the published 4h vertex base) is a raw texture to every reader: the field is then its handle.
    const field = this.present(this.vertexPhiField), texture = this.live(field);
    return texture === field || uniformDetailField(texture) ? texture : field;
  }
  private readonly fieldPages?: UniformTexturePages;
  /** Geometric h fields: base block + patch atlas (uniform-detail-fields.ts). */
  private readonly detailStorage?: UniformDetailStorage;
  private readonly scratchArena?: UniformScratchArena;
  private present(field: GPUTexture): GPUTexture { return this.fieldPages?.publication(field) ?? field; }
  /** The texture a consumer binds: a detail field's current physical texture
   * (replaced when the atlas grows, so resolved at each read). */
  private live(texture: GPUTexture): GPUTexture { return uniformDetailField(texture)?.storage.physical(texture) ?? texture; }
  private readonly executionDenseLevelSetVolumeSource?: DenseLevelSetVolumeConsumerSource;
  private liveSource?: { from: DenseLevelSetVolumeConsumerSource; vertexPhi: GPUTexture; source: DenseLevelSetVolumeConsumerSource };
  get denseLevelSetVolumeSource(): DenseLevelSetVolumeConsumerSource | undefined {
    const band=this.mixedFrame?.narrowBandFlip;
    if(band?.coarseParticles&&band.count>0)return band.surfaceSource;
    // The frame re-creates its presented phi with the h-tile capacity; consumers compare the source object.
    // So does the detail vertex field: named only while the frame holds h-tile capacity.
    if (this.mixedSource && this.mixedFrame) {
      const phi = this.mixedFrame.presentation.phi, detail = this.mixedFrame.ownership.capacity.fineTiles > 0 ? this.mixedSource.vertexPhi : undefined;
      if (this.mixedSource.mixedPressurePhi !== phi || this.mixedSource.detailVertexPhi !== detail)
        this.mixedSource = { ...this.mixedSource, mixedPressurePhi: phi, detailVertexPhi: detail };
    }
    const from = this.mixedSource ?? this.executionDenseLevelSetVolumeSource;
    if (!from || !this.detailStorage) return from;
    const vertexPhi = this.live(from.vertexPhi);
    if (this.liveSource?.from !== from || this.liveSource.vertexPhi !== vertexPhi)
      this.liveSource = { from, vertexPhi, source: vertexPhi === from.vertexPhi ? from : { ...from, vertexPhi, detailVertexPhi: from.detailVertexPhi && vertexPhi, openFraction: this.live(from.openFraction) } };
    return this.liveSource.source;
  }
  private readonly vertexPhiScratch?: GPUTexture;
  /** Most recent transported phi, before closest-point redistancing (diagnostics). */
  get advectedVertexPhiTexture(): GPUTexture | undefined { return this.vertexPhiScratch && this.live(this.present(this.vertexPhiScratch)); }
  private volumeEdges?: GPUBuffer;
  private readonly pageDomain?: UniformPageDomain;
  private readonly nativeRootExecution: boolean = false;
  private readonly volumePageEdge: 0 | 32;
  private readonly volumePageConfig?: UniformVolumePageShaderOptions;
  /** The geometric t=0 surface publication; see encodeInitialPresentationSurface. */
  private initialPublishPipeline?: GPUComputePipeline;
  private readonly initialDetailGroups: () => { authority: GPUBindGroup; publish: GPUBindGroup };
  private readonly shaderSource: string;
  private readonly groupDescriptors = new WeakMap<GPUBindGroup, GPUBindGroupDescriptor>();
  private readonly pressureInputLayout: GPUBindGroupLayout;
  private readonly executionVolumeTexture: GPUTexture;
  get volumeTexture(): GPUTexture { return this.live(this.executionVolumeTexture); }
  get surfaceFieldTexture(): GPUTexture {
    return this.live(this.present(this.surfaceB));
  }
  private readonly executionColumnBaseTexture: GPUTexture;
  get columnBaseTexture(): GPUTexture { return this.executionColumnBaseTexture; }
  private readonly executionVelocityTexture: GPUTexture;
  get velocityTexture(): GPUTexture { return this.live(this.executionVelocityTexture); }
  /** Velocity after advection/forces and before the pressure projection. */
  get preProjectionVelocityTexture(): GPUTexture { return this.live(this.present(this.velocityB)); }
  /** Padded float32 velocity extension retained for opt-in comparison diagnostics. */
  private readonly executionExtrapolatedVelocityTexture?: GPUTexture;
  /** Undefined on Geometric: the mixed extension publishes no padded field. */
  get extrapolatedVelocityTexture(): GPUTexture | undefined { return this.executionExtrapolatedVelocityTexture; }
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
  private readonly executionNegativeBoundaryVelocityBytes: number;
  get negativeBoundaryVelocityBytes(): number { return this.executionNegativeBoundaryVelocityBytes; }
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
  get gridPressureOrigin(): readonly [number, number, number] { return [0, 0, 0]; }
  get physicsFieldsForQA(): {pressure:GPUTexture;gamma:GPUTexture;latticeOrigin:[number,number,number];latticeDimensions:[number,number,number]} {
    return { pressure: this.pressureMultigrid.pressureTexture, gamma: this.live(this.present(this.gammaA)),
      latticeOrigin: [0, 0, 0],
      latticeDimensions: [this.executionInfo.nx + 2, this.executionInfo.ny + 2, this.executionInfo.nz + 2] };
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
  private readonly pressureMultigrid: WebGPUUniformPressureMultigrid;
  // Origin(3), capacity(3), mode(1).
  private pressureWindowHeaderWords = new Uint32Array(7);
  private readonly params: GPUBuffer;
  private readonly solidVoxelScratchOffsetWords: number;
  private readonly solidCutMapOffsetWords: number;
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
  private readonly projectGroup: GPUBindGroup;
  private readonly rigidGroup: GPUBindGroup;
  private readonly reductionGroup: GPUBindGroup;
  private readonly densityTraceGroup: GPUBindGroup;
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
  private sharpeningSweeps: number;
  private surfaceVolumeRounds: number;
  private solidExcessCorrection: boolean;
  private rigidCoupling: boolean;
  private readonly pressureSchedule: UniformCM11aSchedule;
  /** Mixed frames submitted whose receipt is unhandled. */
  private mixedFramesInFlight = 0;
  /** The seeded velocity took its half-step kick (UniformMixedFrame.kick), ahead of the first frame. */
  private mixedKicked = false;
  /** Settles once every mixed frame submitted so far is checked and adopted. */
  private mixedFrameChain: Promise<void> = Promise.resolve();
  /** The GPU relayout is attached (UniformMixedFrame.setRelayout): a frame
   * head runs the census, builder and adopt. Dynamic: every frame, over
   * every tile. Requested and Full: requests only (syncMixedLayout), while
   * they request an h tile. No host layout runs until it is detached. */
  private mixedGpuLayout = false;
  /** Free relayout receipt reads, and the one this frame's head copied into. */
  private readonly mixedRelayoutReads: GPUBuffer[] = [];
  private mixedRelayoutRead?: GPUBuffer;
  /** Diagnostics sequence: an older read never overwrites a newer one. */
  private mixedRelayoutSequence = 0;
  private mixedRelayoutShown = 0;
  /** Rigid bodies on the mixed frame (coupling and census body tiles). */
  private mixedBodies?: UniformMixedBodies;
  /** Compile by need (uniform-pipeline-needs): setup builds the pipelines the
   * initial state can dispatch; a later state waits here for its own while
   * the accepted one keeps advancing. */
  private pipelineNeeds = new UniformPipelineNeeds();
  /** The change that waits (latest wins), replayed once its pipelines exist. */
  private pendingPipelineScene?: SceneDescription;
  private pendingPipelineValues?: MethodParamValues;
  /** Needs a change is waiting for: the "preparing" status. */
  private readonly pipelineWaiting = new Set<UniformPipelineNeed>();
  private pipelineFlight?: Promise<void>;
  private solidFreeWarming = false;
  private pipelineWaitsAccepted = false;
  /** The last advance passed bodies: the next one rebuilds the solid record without them. */
  private mixedHadBodies = false;
  private pressureFrameFailure?: Error;
  private deferredFrameScene?: SceneDescription;
  private deferredFrameValues?: MethodParamValues;
  private deferredFrameBodies?: RigidBodyState[];
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
  /** No lattice cell (halo excluded) set in the static solid voxel mask; see uvSolidFree. */
  private solidVoxelsEmpty = true;
  private readonly prescribedSolidMotion = new UniformPrescribedSolidMotion();
  /** The pinned advance of a fixed-step mode; undefined runs the scene's maxDt. */
  private fixedStep_s: number | undefined;
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
    this.narrowBandFlip = options.narrowBandFlip === true;
    this.narrowBandCoarseParticles = options.narrowBandCoarseParticles === true;
    this.geometricVolume = options.geometricVolume === true;
    this.geometricRedistance = options.geometricRedistance !== false;
    this.volumeDustThreshold = Number.isFinite(options.volumeDustThreshold)
      ? Math.min(1, Math.max(0, options.volumeDustThreshold!)) : 0;
    this.orphanDustThreshold = Number.isFinite(options.orphanDustThreshold)
      ? Math.min(0.05, Math.max(0, options.orphanDustThreshold!)) : 0;
    this.twoLevelVelocity = this.geometricVolume;
    this.surfaceDeficitBalancing = this.geometricVolume && options.surfaceDeficitBalancing === true;
    // The 2D reference runs the same correction: with one cell layer its band,
    // metric and six-tetrahedron fill reduce exactly to the planar algorithm.
    this.totalSurfaceVolume = this.geometricVolume && options.totalSurfaceVolume !== false;
    this.phiCubicAdvection = options.phiCubicAdvection !== false;
    this.phiDrain = options.phiDrain !== false;
    this.phiPreserveSurface = options.phiPreserveSurface === true;
    // Uniform Geometric calls this the SOLVE WINDOW. It needs a positive dust
    // floor for the same reason E3's live set does: the window's diagnostics
    // reduction sums V over the box, which equals the domain sum only while
    // every cell outside the box holds exactly zero, and that is what the
    // floor guarantees. With the floor off the dense schedule is forced.
    this.retainStageDiagnosticsForQA=options.retainStageDiagnosticsForQA===true;
    this.activeRegionEnabled = !this.geometricVolume && options.activeRegion === true
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
    // Geometric: zero skips the stage (the frame never sends a zero band to the GPU).
    this.sharpeningDistance = Math.min(3.1, Math.max(this.geometricVolume ? 0 : 0.1, options.sharpeningDistance ?? 2.1));
    this.sharpeningSweeps = uniformGeometricSharpeningSweeps(options.sharpeningSweeps);
    this.surfaceVolumeRounds = uniformGeometricSurfaceVolumeRounds(options.surfaceVolumeRounds);
    this.solidExcessCorrection = options.solidExcessCorrection !== false;
    this.rigidCoupling = options.rigidCoupling !== false;
    this.pressureSchedule = options.pressureSchedule ?? {
      fullCycles: UNIFORM_CM11A_FULL_CYCLES,
      vCycles: UNIFORM_CM11A_V_CYCLES,
      preSweeps: UNIFORM_CM11A_PRE_SWEEPS,
      postSweeps: UNIFORM_CM11A_POST_SWEEPS,
      residualTolerance: UNIFORM_PRESSURE_RESIDUAL_TOLERANCE,
    };
    this.pressureCycleBudgetLagged = options.pressureCycleBudget !== "fixed";
    this.pressureBudgetHeadroom = Number.isFinite(options.pressureBudgetHeadroom)
      ? Math.round(Math.min(4, Math.max(0, options.pressureBudgetHeadroom!)))
      : this.geometricVolume ? 0 : UNIFORM_CM11A_DEFAULT_BUDGET_HEADROOM;
    this.fixedStep_s = uniformFixedStep_s(options.timeStep);
    this.velocityTransport = options.velocityTransport === "maccormack"
      ? "maccormack" : "semi-lagrangian";
    this.liquidOnlyVelocityAdvection = options.liquidOnlyVelocityAdvection === true;
    // The stage trace's closing marker pass must dispatch observable work, or
    // Metal skips its end-of-pass timestamp and the first sample retires
    // hardware tracing for this solver. Compile it long before the panel asks.
    void GPUStageTimestampRecorder.prepare(device);
    const [nx, ny, nz] = sceneLatticeDimensions(scene, this.geometricVolume ? Number.MAX_SAFE_INTEGER : device.limits.maxTextureDimension3D);
    if (this.geometricVolume) {
      // The mixed-ownership frame owns every geometric advance. Its native
      // fields sit on the all-resident page catalogue; a non-rectangular
      // catalogue falls back to the paged atlas without the scratch arena.
      this.pageDomain = initialUniformPageDomain([nx, ny, nz], UNIFORM_GEOMETRIC_PAGE_EDGE);
      const paged = !uniformPageHasRectangularCoverage(this.pageDomain);
      if (!paged)
        this.scratchArena = new UniformScratchArena(device, [nx, ny, nz], (uniformAbOn("edgebricks") ? uniformBrickCells([nx,ny,nz]) : nx*ny*nz)*UNIFORM_VOLUME_EDGE_BYTES,options.retainStageDiagnosticsForQA,undefined,uniformMixedPressureRootBytes([nx, ny, nz]));
      this.fieldPages = new UniformTexturePages(device, paged, this.scratchArena);
    }
    this.nativeRootExecution = this.fieldPages?.nativeStorage === true;
    this.volumePageEdge = this.geometricVolume ? UNIFORM_GEOMETRIC_PAGE_EDGE : 0;
    const paddedPageCells = this.volumePageEdge ? [nx, ny, nz].reduce((n, d) => n * Math.ceil(d / this.volumePageEdge) * this.volumePageEdge, 1) : 0;
    if (this.geometricVolume) {
      if (Math.max(nx,ny,nz)+2 > device.limits.maxTextureDimension3D)
        throw new Error("Uniform Geometric finest lattice exceeds the device texture limit");
      if (nx*ny*nz*24 > Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize))
        throw new Error("Uniform Geometric donor sums exceed the device storage limit");
    }
    const tileRecords = Math.ceil(nx / 4) * Math.ceil(ny / 4) * Math.ceil(nz / 4);
    // The 4h face table and class map sit in the second conditioning plane,
    // whose per-step clears are ranged away from it. Same size test, plus the
    // shell counter above the two ping-pong planes. Every axis must be a
    // multiple of four: the sampler's convention is "fine face 4t+3 is coarse
    // face t", which is what the extension hierarchy's own transfer computes
    // only when the coarse level tiles the lattice exactly.
    this.twoLevelTileCount = this.geometricVolume
      && [nx, ny, nz].every((value) => value % 4 === 0)
      && UNIFORM_VOLUME_TWO_LEVEL_WORDS_PER_TILE * tileRecords
        + UNIFORM_VOLUME_TWO_LEVEL_COUNTER_WORDS <= nx * ny * nz ? tileRecords : 0;
    // One tile covers the finest trilinear tap, which reaches one cell below a
    // fine tile. The FIM's accurate band is two of the LARGEST cells, so on an
    // anisotropic lattice it spans 2*max(h)/min(h) of the smallest; the shell
    // must contain that too, measured from a seed tile's own boundary.
    const spacing = [scene.container.width_m / nx, scene.container.height_m / ny,
      scene.container.depth_m / nz];
    this.twoLevelShellReach = Math.max(1, Math.ceil(0.5 * Math.max(...spacing) / Math.min(...spacing)));
    this.detail = options.detail ?? uniformDetailSettings({});
    this.mixedCoarsening = this.detail.policy === "dynamic" ? "dynamic" : "regions";
    this.mixedCoarseningReach = Number.isFinite(options.mixedCoarseningReach) ? Math.round(Math.min(8, Math.max(0, options.mixedCoarseningReach!))) : 0;
    this.mixedCoarseningHysteresis = Number.isFinite(options.mixedCoarseningHysteresis) ? Math.round(Math.min(4, Math.max(0, options.mixedCoarseningHysteresis!))) : 0;
    const allocation = planUniformHostAllocation(nx, ny, nz, "maccormack");
    this.executionNegativeBoundaryVelocityBytes = allocation.boundaryVelocityBytes;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const texture3d = (label: string, format: GPUTextureFormat, size: GPUExtent3D, native=false) =>
      this.fieldPages ? this.fieldPages.createTexture({ label, size, dimension: "3d", format, usage },native)
        : device.createTexture({ label, size, dimension: "3d", format, usage });
    const velocity = (label: string) => texture3d(label, "rgba32float", allocation.velocityExtent);
    const scalar = (label: string) => texture3d(label, "r32float", allocation.volumeExtent);
    // Geometric h fields come from the detail storage, registered with the
    // page layer by their logical dims; their bytes are the storage's (refreshMixedAllocation).
    this.detailStorage = this.geometricVolume ? new UniformDetailStorage(device, [nx, ny, nz], uniformDetailStorageOptions(options.detailStorageForQA)) : undefined;
    const field = (label: string, kind: UniformDetailClass, fallback: () => GPUTexture) => this.detailStorage
      ? this.fieldPages!.adopt(this.detailStorage.createField(label, kind, kind === "face" ? "rgba32float" : "r32float"), kind === "vertex" ? [nx + 1, ny + 1, nz + 1] : [nx, ny, nz]) : fallback();
    const hVelocity = (label: string) => field(label, "face", () => velocity(label));
    const hScalar = (label: string) => field(label, "cell", () => scalar(label));
    this.velocityA = hVelocity("Uniform reference velocity A");
    this.velocityB = hVelocity("Uniform reference velocity B");
    const lazyMac = !!this.scratchArena && this.velocityTransport !== "maccormack"
      && !(typeof process !== "undefined" && process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT === "1");
    this.velocityC = lazyMac ? texture3d("Uniform reference unused velocity C", "rgba32float", [1,1,1]) : velocity("Uniform reference velocity C");
    this.velocityD = hVelocity("Uniform reference velocity D");
    const boundaryVelocity = (label: string, size = allocation.boundaryVelocityBytes) => device.createBuffer({ label, size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.boundaryVelocityA = boundaryVelocity("Uniform reference negative boundary velocity A");
    this.boundaryVelocityB = boundaryVelocity("Uniform reference negative boundary velocity B");
    // Read only by legacy (non-mixed) groups: Geometric binds a placeholder.
    this.boundaryVelocityC = boundaryVelocity("Uniform reference negative boundary velocity C", this.geometricVolume ? 16 : allocation.boundaryVelocityBytes);
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
    this.volumeA = hScalar("Uniform reference volume A");
    this.volumeB = hScalar("Uniform reference volume B");
    this.surfaceA = hScalar("Uniform reference smoothed surface A");
    this.surfaceB = hScalar("Uniform reference smoothed surface B");
    this.gammaA = hScalar("Uniform reference transport gamma A");
    this.gammaB = hScalar("Uniform reference transport gamma B");
    if (this.geometricVolume) {
      this.vertexPhiField = field("Uniform Geometric vertex phi", "vertex", () => texture3d("Uniform Geometric vertex phi", "r32float", [nx + 1, ny + 1, nz + 1], true));
      this.vertexPhiScratch = field("Uniform Geometric vertex phi scratch", "vertex", () => texture3d("Uniform Geometric vertex phi scratch", "r32float", [nx + 1, ny + 1, nz + 1], true));
      const edgeBytes = this.volumePageEdge && !this.nativeRootExecution ? paddedPageCells * UNIFORM_VOLUME_EDGE_BYTES : (uniformAbOn("edgebricks") ? uniformBrickCells([nx,ny,nz]) : nx * ny * nz) * UNIFORM_VOLUME_EDGE_BYTES;
      if (edgeBytes > device.limits.maxStorageBufferBindingSize || edgeBytes > device.limits.maxBufferSize)
        throw new Error(`Uniform Geometric receiver stencils require ${edgeBytes} bytes, exceeding the device limit`);
      this.executionDenseLevelSetVolumeSource = { vertexPhi: this.present(this.vertexPhiField), openFraction: this.present(this.gammaB),
        cellSize_m: [scene.container.width_m/nx, scene.container.height_m/ny, scene.container.depth_m/nz] };
      this.volumeEdges = this.scratchArena?.buffer ?? device.createBuffer({ label: "Uniform Geometric nine-donor stencils", size: edgeBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    }
    this.heightA = device.createTexture({ label: "Uniform reference column base", size: [nx, nz], format: "rg32float", usage });
    this.heightB = device.createTexture({ label: "Uniform reference column occupancy", size: [nx, nz], format: "rg32float", usage });
    this.terrainTexture = device.createTexture({ label: "Uniform reference terrain", size: [nx, nz], format: "r32float", usage });
    const transportUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    // The mixed frame reads neither padded transport field: Geometric binds placeholders.
    this.transportA = (this.fieldPages ?? device).createTexture({ label: "Uniform reference transport A", size: this.geometricVolume ? [1,1,1] : allocation.transportExtent, dimension: "3d", format: "rgba32float", usage: transportUsage });
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
    this.params = device.createBuffer({ label: "Uniform reference parameters", size: 272, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.solidMask = new SolidOccupancyMask([nx, ny, nz]);
    this.solidMask.update(solidWorldForScene(scene));
    this.solidVoxelsEmpty = uniformSolidMaskEmpty(this.solidMask);
    const packedSolidVoxels = this.solidMask.words;
    const activeRegionBytes = UNIFORM_ACTIVE_HEADER_WORDS * 4;
    this.activeRegion = device.createBuffer({
      label: "Uniform reference active liquid region", size: activeRegionBytes+(this.pageDomain?.words.byteLength??0),
      // readStats copies the published bounds/counters into its readback
      // packet in both dense and sparse modes.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const activeSummaryCount = Math.ceil(nx / 4) * Math.ceil(ny / 4) * Math.ceil(nz / 4);
    const activeSummaryBytes = activeSummaryCount * UNIFORM_ACTIVE_SUMMARY_BYTES;
    this.solidVoxelScratchOffsetWords = (activeRegionBytes + activeSummaryBytes) / 4;
    // The mixed solid library's tile cut map follows the mask (see
    // UniformMixedSolidResources.cutMapOffsetWords); all cut until built.
    this.solidCutMapOffsetWords = Math.ceil((this.solidVoxelScratchOffsetWords + packedSolidVoxels.length) / 64) * 64;
    this.activeScratch = device.createBuffer({
      label: "Uniform reference active liquid census scratch and summaries",
      size: 4 * (this.solidCutMapOffsetWords + activeSummaryCount),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.activeScratch, this.solidVoxelScratchOffsetWords * 4,
      packedSolidVoxels.buffer as ArrayBuffer, packedSolidVoxels.byteOffset,
      packedSolidVoxels.byteLength);
    device.queue.writeBuffer(this.activeScratch, 4 * this.solidCutMapOffsetWords, new Uint32Array(activeSummaryCount).fill(1));
    this.activeDispatch = device.createBuffer({
      label: "Uniform reference active indirect dispatches", size: activeRegionBytes,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    if(this.pageDomain) device.queue.writeBuffer(this.activeRegion,activeRegionBytes,this.pageDomain.words.buffer);
    // Created before the extrapolator: the extension binds it read_write to
    // read the tile classes and to publish the 4h face table the sampler reads.
    const balanceRecords = this.pageDomain && !this.nativeRootExecution ? this.pageDomain.capacity * (this.pageDomain.edge / 4) ** 3 : tileRecords;
    const balanceChunks = this.geometricVolume && balanceRecords > 1024
      && options.systemBuildForQA !== "baseline" && uniformAbOn("balancetree")
      ? Math.ceil(balanceRecords / 1024) : 0;
    this.surfaceDeficitBalanceBytes = this.geometricVolume ? (2 + 2 * balanceRecords + 2 * balanceChunks) * 4 : 0;
    const pageBaseBytes = (this.scratchArena?.conditioningBytes ?? allocation.conditioningBytes) + this.surfaceDeficitBalanceBytes;
    if (this.volumePageEdge) this.volumePageConfig = {
      edge: this.volumePageEdge, base: pageBaseBytes / 4,
      nativeRecords: this.nativeRootExecution,
      work: Math.max(nx,ny,nz)>64,
      count: [nx, ny, nz].reduce((n, d) => n * Math.ceil(d / this.volumePageEdge), 1),
    };
    // After the work lists: the surface-deficit and sharpening-classify tile
    // boxes (balancesupport, classifysupport), then the phi census box and one
    // static solid class per 4h tile (seedwindow).
    const pageBytes = this.volumePageConfig ? 4 * (this.volumePageConfig.work
      ? uniformVolumeWorkLayout(this.volumePageConfig.base,this.volumePageConfig.count,tileRecords).words + 18 + tileRecords
      : 8 + 2 * this.volumePageConfig.count) : 0;
    const fullLattice = !this.activeRegionEnabled && uniformAbOn("staticid");
    const source = this.geometricVolume ? createUniformReferenceComputeShader(true, 3, this.volumePageConfig,this.nativeRootExecution ? undefined : this.pageDomain, fullLattice) : uniformReferenceComputeShader;
    const fixedFields = new Map([
      [0,this.velocityA],[1,this.velocityB],[3,this.pressureB],[4,this.volumeA],[5,this.volumeB],
      [12,this.velocityC],[13,this.velocityD],[14,this.transportA],[16,this.surfaceA],
      [20,this.surfaceA],[24,this.gammaA],[25,this.gammaB],[31,this.vertexPhiField!],[32,this.vertexPhiScratch!],
    ]);
    // These bindings always hold retained native textures, even when ping-pong
    // groups change. Keep their interpolation free of scratch address branches.
    const nativeBindings = new Set([0,1,2,3,4,5,12,13,14,16,20,21,24,25,31,32]);
    this.shaderSource = this.fieldPages?.shader(source,fixedFields,!!this.pageDomain && !this.nativeRootExecution,this.nativeRootExecution,false,nativeBindings) ?? source;
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
      options.sourceAwareExtension ?? this.geometricVolume, options.fuseExtensionPack, this.fieldPages, this.nativeRootExecution ? undefined : this.pageDomain, undefined, options.extensionWorkForQA === "baseline", this.geometricVolume,
    );
    // Without a ceil(n/4) hierarchy level there is no 4h field to sample.
    if (!this.velocityExtrapolator.coarseVelocityTableAvailable) this.twoLevelTileCount = 0;
    if (this.twoLevelTileCount === 0) this.twoLevelVelocity = false;
    this.extrapolationActiveFrontPassCeiling = this.velocityExtrapolator.activeFrontPassCeiling;
    if (options.extensionFrontSweeps !== undefined) this.velocityExtrapolator.setFrontPasses(options.extensionFrontSweeps);
    this.reductions = device.createBuffer({ label: "Uniform reference diagnostics and volume control", size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
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
    const initialRigidBodies = initializeRigidBodies(scene.rigidBodies);
    this.prescribedSolidMotion.sample(initialRigidBodies, 0);
    this.rigidSystem.syncBodies(initialRigidBodies);
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
      undefined, false, 3, options.pressureCycleDispatch === "indirect" ||
        (options.pressureCycleDispatch !== "direct" && pagedPressure),
      pagedPressure, options.pressureStorageForQA === "paged-logical",
      uniformAbOn("inplace")
        && scene.container.depthBoundary !== "symmetry", this.scratchArena && !pagedPressure ? this.fieldPages : undefined, this.geometricVolume && options.pressureSmoothingForQA !== "dense", options.pressureAuthorityForQA !== "raw", options.systemBuildForQA !== "baseline", this.geometricVolume, this.geometricVolume, this.params);
    this.mainPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.mainLayout] });
    const sampler = device.createSampler({ minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    let physicalViews = false;
    const view = (texture: GPUTexture) => this.fieldPages?.view(texture, physicalViews) ?? texture.createView();
    const group = (velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture,
      pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture,
      heightIn: GPUTexture, heightOut: GPUTexture, predicted = velocityIn,
      reversed = velocityIn, transport = this.transportA, surface = volumeIn,
      gammaRead = this.gammaA, gammaWrite = this.gammaB,
      boundaryRead = this.boundaryVelocityA, boundaryWrite = this.boundaryVelocityB,
      velocityPhase = volumeIn, reversePhi = false, pressureOnly = false) => this.createPageAwareGroup({
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
          { binding: 10, resource: { buffer: this.rigidSystem.stateBuffer } }, { binding: 11, resource: { buffer: this.rigidExchange } },
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
      });
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
    this.projectGroup = group(this.velocityB, this.velocityA, this.pressureMultigrid.pressureTexture, this.pressureA, this.volumeB, this.volumeA, this.heightB, this.heightA, this.velocityB, sharedFaceOpen, this.transportA, this.volumeB, this.gammaA, this.gammaB, this.boundaryVelocityB, this.boundaryVelocityA);
    this.rigidGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityA, this.velocityA, this.transportA, this.volumeA, this.gammaA, this.gammaB, this.boundaryVelocityA, this.boundaryVelocityB);
    this.reductionGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.densityTraceGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA,
      this.velocityA, this.velocityA, this.transportA, this.volumeA, this.gammaA, this.gammaB);
    // Paper Sec. 3.4 step 7 scatters the deficit from pre-advection gamma^n.
    this.densityScatterGroup = group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA,
      this.velocityA, this.velocityA, this.transportA, this.volumeA, this.gammaA, this.gammaB);
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
    // Packed detail storage: the two t=0 kernels' groups over the fields'
    // physical textures, built at each use (the atlas may have grown).
    this.initialDetailGroups = () => {
      physicalViews = true;
      try {
        return {
          authority: group(this.velocityA, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.surfaceA, this.heightB, this.heightA),
          publish: group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.surfaceB, this.heightB, this.heightA),
        };
      } finally { physicalViews = false; }
    };

    const count = nx * ny * nz;
    // The legacy host plan reserves nine scalars; this owner has eight. Add
    // the extension hierarchy/origins and the two RG columns plus terrain,
    // which that plan does not include. Scratch backing is reconciled below.
    const hostAuxiliaryBytes = this.velocityExtrapolator.scratchBytes
      - (nx+2)*(ny+2)*(nz+2)*6*16 - count*4 + nx*nz*20;
    this.executionInfo = {
      nx, ny, nz, storedNy: ny, cellCount: count, equivalentUniformCells: count,
      compressionRatio: 1, activeCompressionRatio: 1, activeSampleCount: count,
      regularLayers: ny, maximumNeighborDelta: 0, gridKind: "uniform",
      cellSize_m: Math.min(scene.container.width_m / nx, scene.container.height_m / ny, scene.container.depth_m / nz),
      pressureIterations: 0, pressureSolver: `CM11a ${pagedPressure ? "paged (QA)" : this.pageDomain ? "native-page" : "dense"} LCP multigrid (${this.pressureSchedule.fullCycles} Full-Cycles + ${this.pressureSchedule.vCycles} V-Cycles, ${this.pressureSchedule.preSweeps}/${this.pressureSchedule.postSweeps} pre/post ${this.geometricVolume ? "projected Jacobi; V-first adaptive" : "PRBGS"})`,
      allocatedBytes: allocation.allocatedBytes - (this.geometricVolume ? count*8-8 : 0) - (this.scratchArena ? allocation.conditioningBytes-this.scratchArena.conditioningBytes : 0) - (lazyMac ? nx*ny*nz*16+(nx+2)*(ny+2)*(nz+2)*16-32 : 0) - (this.geometricVolume ? (nx+2)*(ny+2)*(nz+2)*16-16+allocation.boundaryVelocityBytes-16 : 0) + 8 + pageBytes + this.surfaceDeficitBalanceBytes + this.pressureMultigrid.allocatedBytes
        + (this.geometricVolume ? 8 * (nx+1)*(ny+1)*(nz+1) + (this.scratchArena ? 0 : count*24 + (this.volumeEdges?.size ?? 0)) + 12 : 0)
        + activeRegionBytes * 3 + (this.pageDomain ? this.pageDomain.words.byteLength : 0) + activeSummaryBytes + packedSolidVoxels.byteLength
        + hostAuxiliaryBytes + (this.symmetryStageAuditMacCormackBuffer ? 0 : 16), quality,
      submittedTime_s: 0, simulatedTime_s: 0, completedTime_s: 0,
      simulationLag_s: 0, encodedSteps: 0, maximumTallCellHeight: 0,
      volumeControl: !this.narrowBandFlip,
      hostFluidAuthority: "gpu-resident", hostSimulationSizedWorkItems: 0,
      uniformPressureCycleBudget: this.geometricVolume ? "adaptive" : this.pressureCycleBudgetLagged ? "lagged" : "fixed",
      hostSchedulingUsesReadback: this.pressureCycleBudgetLagged && this.pressureMultigrid.residualTolerance > 0,
      ...(this.pageDomain?{uniformDomainAuthority:"pages" as const,uniformDomainPages:this.pageDomain.count,
        uniformDomainMigration:pagedPressure ? "Paged fluid and pressure fields (QA); all-resident domain" : this.nativeRootExecution ? "Native rectangular fields; all-resident page catalogue" : "Paged fluid fields; native pressure workspace; all-resident domain"}:{}),
      ...(this.volumePageConfig ? { uniformVolumePageEdge: this.volumePageEdge, uniformVolumePagesTotal: this.volumePageConfig.count, uniformVolumePageBytes: this.scratchArena?.edgeBytes ?? this.volumeEdges!.size } : {}),
    };
    this.executionVolumeTexture = this.present(this.volumeA);
    this.executionColumnBaseTexture = this.heightA;
    this.executionVelocityTexture = this.present(this.velocityA);
    // What the overlays and the particles bind (volumeTexture, velocityTexture, preProjectionVelocityTexture, surfaceFieldTexture, the open fraction).
    if (!this.geometricVolume) this.executionExtrapolatedVelocityTexture = this.present(this.transportA);
    if(!this.scratchArena){
      this.present(this.velocityB);
      this.present(this.velocityExtrapolator.activeStateTexture);
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
      if(options.geometricVolume) await runner.run([{id:"uniform.mixed",phase:"solver-pipelines",label:"Prepare shared mixed ownership",run:()=>solver!.initializeMixedFrame()}]);
      return solver!;
    } catch (error) {
      solver?.destroy();
      throw error;
    }
  }

  private mixedFrame?: UniformMixedFrame;
  private mixedFrameTrace(): UniformMixedFrameTrace | undefined {
    const instrumentation=usePerformanceInstrumentationStore.getState(),requested=performance.now();
    if(!instrumentation.enabled||this.physicsTracePending||requested-this.lastPhysicsTraceAt_ms<UNIFORM_PHYSICS_TRACE_CADENCE_MS)return undefined;
    this.executionInfo.physicsCPUTrace=undefined;
    if(!GPUStageTimestampRecorder.supported(this.device)||!GPUStageTimestampRecorder.markersReady(this.device)){
      this.executionInfo.physicsTrace=undefined;
      this.executionInfo.physicsTraceUnavailable="GPU hardware timing is unavailable. No substitute timing is reported.";
      return undefined;
    }
    const sampleId=++this.physicsTraceSampleId,context=`uniform:sim-${this.lastTime.toFixed(6)}`;
    // Close each submitted command buffer before awaiting its receipt. A
    // timestamp chain spanning submissions would charge CPU waits to pressure.
    const segments:GPUStageTimestampRecorder[]=[];
    let active:GPUStageTimestampRecorder|undefined;
    this.physicsTracePending=true;this.lastPhysicsTraceAt_ms=requested;
    return {
      instrument:encoder=>{
        if(active)throw new Error("Mixed timing segment already open");
        active=new GPUStageTimestampRecorder(this.device,sampleId,"physics",context);active.begin();
        return active.instrument(encoder);
      },
      phase:(encoder,phase)=>active!.completePhase(encoder,phase),
      submit:(encoder,anchor)=>{
        active!.anchorFinalBoundary(anchor);active!.resolve(encoder);segments.push(active!);active=undefined;
      },
      submitted:()=>{
        this.executionInfo.physicsCaptureIdentity={sampleId,context,frameId:gpuPhysicsPerformanceActivityFrameId({sampleId,context})};
        void Promise.all(segments.map(segment=>segment.read())).then(traces=>{
          const current=usePerformanceInstrumentationStore.getState();
          if(this.disposed||!current.enabled||current.enabledAt_ms>requested)return;
          if(traces.some(trace=>!trace)){
            this.executionInfo.physicsTrace=undefined;
            this.executionInfo.physicsTraceUnavailable="GPU hardware timing was invalid. No substitute timing is reported.";
            return;
          }
          const complete=traces.map(trace=>trace!);
          this.executionInfo.physicsTrace={...complete.at(-1)!,total_ms:complete.reduce((sum,trace)=>sum+trace.total_ms,0),phases:complete.flatMap(trace=>trace.phases)};
          this.executionInfo.physicsTraceUnavailable=undefined;
        }).catch(error=>{
          if(!this.disposed){this.executionInfo.physicsTrace=undefined;this.executionInfo.physicsTraceUnavailable=`GPU hardware timing failed: ${String(error)}`;}
        }).finally(()=>{this.physicsTracePending=false;});
      },
      abort:()=>{active?.destroy();segments.forEach(segment=>segment.destroy());this.physicsTracePending=false;},
    };
  }
  private mixedDiagnostics?: UniformMixedDiagnostics;
  private mixedAccountedBytes?:number;
  /** Detail storage bytes in executionInfo: its fields' physical textures, which grow with residency. */
  private detailAccountedBytes=0;
  private refreshMixedAllocation():void{
    if(!this.mixedFrame||this.mixedAccountedBytes===undefined)return;
    const bytes=this.mixedFrame.allocatedBytes,detail=this.detailStorage?.allocatedBytes??0;
    this.executionInfo.allocatedBytes+=bytes-this.mixedAccountedBytes+detail-this.detailAccountedBytes;this.mixedAccountedBytes=bytes;this.detailAccountedBytes=detail;
  }
  /** Layout identity of the applied host layout. */
  private mixedRegionKey="";
  /** Identity of the last request the frame refused (its h tiles exceed what
   * the device holds): the accepted generation above keeps running. */
  private mixedRefusedKey="";
  private mixedGeneration=0;
  /** Terrain heights in cells, retained for solid promotion. */
  private mixedTerrainCells?: Float32Array;
  private mixedSolidMaskStamp=0;
  /** The promotion of the mask at this stamp: a scan of the lattice, made again only for another mask. */
  private mixedSolidTiles?:{stamp:number;terrain?:Float32Array;tiles:ReturnType<typeof uniformMixedSolidTiles>&{any:boolean}};
  private mixedSolidPromotion(){
    const terrain=sceneHasTerrain(this.scene)?this.mixedTerrainCells:undefined,held=this.mixedSolidTiles;
    if(held&&held.stamp===this.mixedSolidMaskStamp&&held.terrain===terrain)return held.tiles;
    const solid=uniformMixedSolidTiles(refinementRegionLattice(this.scene).dimensions as [number,number,number],this.solidMask.words,SOLID_OCCUPANCY_MASK_HEADER_WORDS,terrain);
    const tiles={...solid,any:solid.coupled.some(c=>c!==0)};
    this.mixedSolidTiles={stamp:this.mixedSolidMaskStamp,terrain,tiles};
    return tiles;
  }
  private mixedSource?: DenseLevelSetVolumeConsumerSource;
  private async initializeMixedFrame():Promise<void>{
    this.assertMixedOptions();
    if(!this.scratchArena||!this.vertexPhiField||!this.vertexPhiScratch)throw new Error("Mixed Uniform requires the shared native field arena");
    const fine=createUniformMixedLayout(refinementRegionLattice(this.scene),[]);
    const promotion=this.mixedSolidPromotion();
    this.mixedSolidWords=this.solidMask.words.slice();
    const detail=this.detailStorage!;
    const pressureGeometry={target:detail.createField("Uniform mixed pressure surface target","cell","r32float",true),centerPhi:detail.createField("Uniform mixed pressure centre phi","cell","r32float",true)};
    this.mixedPressureGeometry=pressureGeometry;
    // Host layouts (Requested, Full) reserve their own h tiles, starting from
    // none; Dynamic holds every tile until its census is bounded. No byte
    // budget is passed: WebGPU reports no device memory, so only the
    // device's buffer limits refuse a reservation (UniformMixedFrameCapacity).
    const tiles=fine.tiles.length,fixed=uniformMixedFixedFineCapacity(tiles),gpuLayout=this.detail.policy==="dynamic";
    // Setup builds exactly what the initial state can dispatch: the scene's
    // solids, its bodies as the roster, the constructed detail settings.
    const solids=!this.solidVoxelsEmpty||sceneHasTerrain(this.scene),bodies=this.scene.rigidBodies.length;
    const needs=this.pipelineNeeds=new UniformPipelineNeeds([...(solids||bodies?[]:["solidFree"] as const),...this.pipelineNeedsOf({solids,bodies})]);
    // The h store spans the lattice unless the host bounds the h tiles: its
    // pipelines then start in the direct form (UniformDetailStorage.prefer).
    detail.prefer(this.detail.policy!=="requested"||bodies>0);
    this.mixedFrame=new UniformMixedFrame(this.device,fine,{fineTiles:fixed??(gpuLayout?tiles:0),liquidBand:gpuLayout,fixed:fixed!==undefined},{
      narrowBandFlip:this.narrowBandFlip,narrowBandCoarseParticles:this.narrowBandCoarseParticles,arena:this.scratchArena,volume:this.volumeA,volumeScratch:this.volumeB,
      velocity:this.velocityA,velocityScratch:this.velocityB,departure:this.velocityD,
      negative:this.boundaryVelocityA,negativeScratch:this.boundaryVelocityB,negativeDeparture:this.boundaryVelocityD,
      phi:this.vertexPhiField,phiScratch:this.vertexPhiScratch,phase:this.surfaceA,centerPhi:this.surfaceB,target:this.gammaB,correction:this.gammaA,
      pressure:this.pressureMultigrid.prepareMixedContinuation(),extension:this.velocityExtrapolator.prepareMixedContinuation(),uniformGroup:this.pressureMultigridGroup,sourceParams:this.params,
      // Always solid-coupled, even with no cut cells: a live voxel edit needs
      // the solid kernels without rebuilding the frame. Each stage has a
      // solid-free twin (umSolidsPresent=0), selected per advance while the
      // scene has no interior voxel, terrain or body; a twin is built when
      // its state is first held, so the first voxel waits for a compile.
      solid:{params:this.params,scratch:this.activeScratch,terrain:this.terrainTexture,bodies:this.rigidSystem.stateBuffer,coupledTiles:promotion.coupled.reduce((n,c)=>n+c,0),cutMapOffsetWords:this.solidCutMapOffsetWords,paramsWritten:()=>this.paramsWritten},
      pressureGeometry,detail,needs,
    },this.scene.container.top==="open",this.pressureSchedule);
    this.mixedFrame.solid!.present=solids||bodies>0;
    this.mixedDiagnostics=new UniformMixedDiagnostics(this.device,this.mixedFrame.ownership,this.volumeA,this.velocityA,this.vertexPhiField,this.reductions);
    // The scene uniform: the census marks this frame's drop and inflow plug band on the GPU.
    this.mixedDynamic=new UniformMixedDynamicClassifier(this.device,this.mixedFrame.ownership,this.volumeA,this.vertexPhiField,this.velocityB,this.params,this.mixedFrame.heldDistance);
    this.mixedBuilder=new UniformMixedLayoutBuilder(this.device,this.mixedDynamic.bandBits,this.mixedFrame.ownership);
    this.mixedBodies=new UniformMixedBodies(this.device,this.mixedFrame.ownership,this.mixedFrame.solid!,{velocity:this.velocityA,phi:this.vertexPhiField,exchange:this.rigidExchange});
    // One concurrent compile: serial stages left a first load waiting on
    // each uncached pipeline in turn.
    const dynamic=this.mixedDynamic,builder=this.mixedBuilder,mixedBodies=this.mixedBodies;
    await Promise.all([this.mixedFrame.initialize(),this.mixedDiagnostics.initialize(),
      needs.declare(["dynamic"],()=>Promise.all([dynamic.initialize(),builder.initialize()])),
      needs.declare(["bodies"],()=>mixedBodies.initialize())]);
    this.mixedFrame.setLayoutViews(this.layoutViews);
    this.mixedAccountedBytes=this.mixedFrame.allocatedBytes;
    this.executionInfo.allocatedBytes+=this.mixedAccountedBytes+this.mixedDynamic.allocatedBytes+this.mixedBuilder.allocatedBytes+(this.mixedBodies?.allocatedBytes??0);
    this.mixedSource={vertexPhi:this.vertexPhiField,coarseVertexPhi:this.mixedFrame.coarseVertexPhi,detailVertexPhi:this.mixedFrame.ownership.capacity.fineTiles>0?this.vertexPhiField:undefined,openFraction:this.gammaB,cellSize_m:fine.lattice.cellSize_m,mixedOwnership:this.mixedFrame.ownership.presentation,mixedPressure:this.mixedFrame.presentation.pressure,mixedPressurePhi:this.mixedFrame.presentation.phi,mixedSupport:{buffer:this.mixedFrame.ownership.support}};
    for(let i=0;i<UNIFORM_MIXED_RELAYOUT_READS;i++)this.mixedRelayoutReads.push(this.device.createBuffer({label:`Uniform mixed relayout receipt read ${i}`,size:UNIFORM_MIXED_RELAYOUT_RECEIPT.words*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}));
    this.executionInfo.allocatedBytes+=UNIFORM_MIXED_RELAYOUT_READS*UNIFORM_MIXED_RELAYOUT_RECEIPT.words*4;
    // The initial layout is the host's (authored regions, solids and bodies
    // at h); dynamic coarsening then relayouts it on the GPU every frame.
    // Accessor detail storage holds the t=0 uploads until its first layout is
    // installed, so the t=0 authority and publication follow that layout.
    // They also make the first write of the parameter block, which the first
    // layout already reads: UniformMixedFrame.updateLayout builds the all-4h
    // solid record and the tile cut map from it. Written here, as identity's
    // t=0 publication has by now; built from a zero block the record holds
    // every tile uncut for the life of the frame.
    const deferred=this.detailStorage?.layout.placement==="packed";
    if(deferred)this.writeParams(0,Math.min(this.scene.rigidBodies.length,12),0);
    this.syncMixedLayout();
    // t=0: the uploaded phi on the initial layout (a refused or unchanged
    // layout encodes no relayout publish).
    this.mixedFrame.publishCoarseVertexPhi();
    if(deferred)this.encodeInitialPresentationSurface();
  }
  private mixedDynamic?: UniformMixedDynamicClassifier;
  /** GPU ownership builder for dynamic relayouts; region edits stay on the CPU path. */
  private mixedBuilder?: UniformMixedLayoutBuilder;
  private mixedBuilderStaticKey?: string;
  /** A requests-only layout's request (Requested, Full): the detail plan, the
   * host's bound on its h tiles (the plan's and, with solid contact
   * requested, every tile a solid can promote: the census's
   * liquid-conditional promotion is a subset) and the census's solid mask. */
  private mixedRequest?:{
    /** The requests' tiles (the plan of a manual-only list), moved by difference. */
    tiles:UniformDetailRequestTiles;
    /** Per tile, bit 0: a request holds it; bit 1: a solid can promote it. */
    mask:Uint8Array;count:number;coupled?:Uint8Array;
    /** What it answers: set again, a candidate that was not accepted returns to it. */
    input:{requests:readonly DetailRequest[];solid?:ReturnType<WebGPUUniformReferenceSolver["mixedSolidPromotion"]>};
    /** Tiles whose request bit moved since the builder's statics were written from `tiles`. */
    moved:number[];
    /** The last set moved request tiles only: their count, and with their
     * 26 neighbours, each once (UniformWorkEdit). Absent: the solid mask moved too. */
    edit?:UniformWorkEdit};
  /** The request tiles the builder's static masks mirror (editStatic follows them by `moved`). */
  private mixedStaticsOf?:UniformDetailRequestTiles;
  /** The coupled mask the census holds (setSolid). */
  private mixedSolidSet?:Uint8Array;
  private mixedNoSolid?:Uint8Array;
  /** The build that is due answers request edits only (mixedRequest.edit,
   * summed): the frame keeps its launch budgets and adds this bound.
   * Absent: any other cause, and the budgets start again at the ceilings. */
  private mixedEdit?:UniformWorkEdit;
  /** The h tiles of the last host layout (hostMixedLayout): zero is all-4h, the layout a request's attach edits. */
  private mixedHostFine=-1;
  private mixedEditSeen?:{stamp:number;tiles:Uint32Array};
  /** The layout is due a build. edit: a request edit's bound; any cause without one makes the build an unknown one. */
  private staleMixed(edit?:UniformWorkEdit):void{
    const held=this.mixedEdit;
    this.mixedEdit=!edit||(this.mixedStale&&!held)?undefined:{tiles:(held?.tiles??0)+edit.tiles,reach:(held?.reach??0)+edit.reach};
    this.mixedStale=true;
  }
  /** Tiles within Chebyshev distance one of the listed tiles, each once. */
  private mixedEditReach(dimensions:readonly [number,number,number],...lists:readonly number[][]):number{
    const [tx,ty,tz]=dimensions,n=tx*ty*tz,total=lists.reduce((sum,l)=>sum+l.length,0);
    if(27*total>=n)return n;
    let seen=this.mixedEditSeen;
    if(!seen||seen.tiles.length!==n||seen.stamp===0xffffffff)seen=this.mixedEditSeen={stamp:0,tiles:new Uint32Array(n)};
    const stamp=++seen.stamp,tiles=seen.tiles;let count=0;
    for(const list of lists)for(const t of list){
      const x=t%tx,y=Math.floor(t/tx)%ty,z=Math.floor(t/(tx*ty));
      for(let k=Math.max(0,z-1);k<=Math.min(tz-1,z+1);k++)for(let j=Math.max(0,y-1);j<=Math.min(ty-1,y+1);j++)for(let i=Math.max(0,x-1);i<=Math.min(tx-1,x+1);i++){
        const q=i+tx*(j+ty*k);if(tiles[q]!==stamp){tiles[q]=stamp;count++;}
      }
    }
    return count;
  }
  /** The attached relayout is Dynamic's census: the importance criteria on,
   * the pressure band at its liquid bound. */
  private mixedCensus=false;
  /** The builder's admission capacity: the h tiles the storage holds, or a
   * staged return under them (mixedCap). undefined: detached. */
  private mixedReserved?:number;
  /** The largest need a deferred build reported that the storage does not hold yet. */
  private mixedDeferredNeed=0;
  /** The capacity rule's view of this request (followMixedCapacity). needs:
   * the h tiles its last builds asked for, oldest first, the rule's window
   * at most. quiet: receipts since the capacity last grew. hinted: no
   * receipt of it has arrived, so the host's own bound holds the capacity. */
  private mixedNeeds:number[]=[];
  private mixedQuiet=0;
  private mixedHinted=true;
  /** A staged return: builds from `build` on are admitted under `tiles`. A
   * receipt of one that fit confirms no later generation holds more, and
   * the storage follows; one that did not takes the cap back. */
  private mixedCap?:{tiles:number;build:number;confirmed?:boolean};
  /** Builds encoded; the receipts read back from them, for the next sync. */
  private mixedBuilds=0;
  private mixedReceipts:{build:number;epoch:number;sync:number;need:number;admit:number;fine:number}[]=[];
  private mixedSyncs=0;
  /** Static h tiles the builder was given that no receipt counts yet (Dynamic: a region, a moved focus). */
  private mixedStaticsAdded?:{tiles:number;build:number};
  private readonly mixedCapacityRule=uniformDetailCapacityRule();
  private mixedDeferrals={builds:0,tiles:0,changes:0};
  /** Bumped when the request changes: an older receipt's need is another request's. */
  private mixedRequestEpoch=0;
  /** A requests-only census must run every frame: bodies, or solid contact over solids. */
  private mixedLive=false;
  /** The next head must build: the request, the capacity or the roster changed since the last build. */
  private mixedStale=false;
  private mixedRosterKey="";
  /** The last advance's roster: a sync between frames bounds the same bodies. */
  private mixedRoster:readonly RigidBodyState[]=[];
  private mixedRosterDt=0;
  /** What the last accepted sync answered: the scene, the settings and the solid mask. */
  private mixedSynced?:{scene:SceneDescription;detail:UniformDetailSettings;stamp:number};
  /** A live solid edit's tiles under Requested or Full, joined to the next
   * census: at h for the build whose head displaces liquid out of the new
   * solid (the displacement moves h owners only), then whatever the requests
   * or the census say. build: the build that carried it. Its receipt lets
   * the tiles go when it was admitted; deferred, the join and the
   * displacement are made again for the next build (takeMixedReceipts), so
   * no edit's displacement is retired before its tiles exist. */
  private mixedEditJoin?:{tiles:Uint8Array;build?:number};
  /** The h store's window (UniformDetailStorage.window). Oldest first: the
   * tile box (low inclusive, high exclusive) that bounds the h tiles of each
   * request since the last build a receipt showed adopted (none: any tile),
   * with the first build that sees it. Their union is the store's box. */
  private mixedWindows:{box?:readonly number[];build:number}[]=[];
  private mixedWindowOf?:{input:unknown;join?:Uint8Array;box?:readonly number[]};
  private readonly mixedTileBoxes=new WeakMap<Uint8Array,readonly number[]|null>();
  /** A tile mask's box, scanned once per mask; null: no tile. */
  private mixedTileBox(tiles:Uint8Array):readonly number[]|null{
    let box=this.mixedTileBoxes.get(tiles);if(box!==undefined)return box;
    const [tx,ty,tz]=refinementRegionLattice(this.scene).dimensions.map(d=>d/4) as [number,number,number],b=[tx,ty,tz,0,0,0];
    for(let z=0,t=0;z<tz;z++)for(let y=0;y<ty;y++)for(let x=0;x<tx;x++,t++){
      if(!tiles[t])continue;
      if(x<b[0]!)b[0]=x;if(y<b[1]!)b[1]=y;if(z<b[2]!)b[2]=z;if(x>=b[3]!)b[3]=x+1;if(y>=b[4]!)b[4]=y+1;if(z>=b[5]!)b[5]=z+1;
    }
    box=b[3]?b:null;this.mixedTileBoxes.set(tiles,box);return box;
  }
  /** Before a reserve that may admit: the storage's window is the request's
   * tiles (its boxes, the solid-contact promotion's tiles, a live edit's
   * join) and those of every request a build may still hold. any: the h
   * tiles are not the host's to bound (Dynamic's census, bodies). first: the
   * t=0 kernels address the lattice, so the first store spans it. */
  private windowMixed(any:boolean,first=false):void{
    const storage=this.detailStorage;if(!storage?.layout.domain)return;
    const queue=this.mixedWindows,request=this.mixedRequest;let box:readonly number[]|undefined;
    if(!any&&request){
      const join=this.mixedEditJoin?.tiles;let held=this.mixedWindowOf;
      if(!held||held.input!==request.input||held.join!==join){
        const lattice=refinementRegionLattice(this.scene);let b:number[]|undefined;
        const add=(o:readonly number[]|null|undefined)=>{if(o)b=b?[0,1,2].map(a=>Math.min(b![a]!,o[a]!)).concat([3,4,5].map(a=>Math.max(b![a]!,o[a]!))):[...o];};
        for(const r of request.input.requests){const o=uniformDetailTileBounds(lattice,r.bounds_m);if(o)add([...o.min,...o.max]);}
        const forced=request.input.solid?.forced;if(forced)add(this.mixedTileBox(forced));
        if(join)add(this.mixedTileBox(join));
        this.mixedWindowOf=held={input:request.input,join,box:b};
      }
      box=held.box;
    }
    if(first)queue.push({build:this.mixedBuilds});
    const last=queue[queue.length-1];
    if(!last||last.box?.join()!==box?.join())queue.push({box,build:this.mixedBuilds});
    let union:number[]|undefined;
    for(const e of queue){
      if(!e.box){union=undefined;break;}
      const o=e.box,u:number[]|undefined=union;union=u?[0,1,2].map(a=>Math.min(u[a]!,o[a]!)).concat([3,4,5].map(a=>Math.max(u[a]!,o[a]!))):[...o];
    }
    storage.window(union);
  }
  /** A host (CPU) layout, between frames. Three uses remain: the t = 0
   * layout (the uploads are h data, deposited once on it), zero detail (no
   * relayout is attached), and the return of h-tile capacity (a GPU-built
   * generation's h tiles reach the host two frames late, so it cannot
   * shrink under one). mask: the tiles at h on a 4h background; absent,
   * Dynamic's start (Coarse regions at 4h, solid contact at h). Returns the
   * frame's refusal, with nothing changed. */
  private hostMixedLayout(mask?:Uint8Array):string|undefined{
    const started=performance.now(),lattice=refinementRegionLattice(this.scene),regions=this.scene.fluid.refinementRegions??[];
    let layout;
    if(mask){
      const snapped=regions.flatMap(r=>{const box=uniformDetailTileBounds(lattice,{min:r.min_m,max:r.max_m});return box?[{id:r.id,...box}]:[];});
      layout=createUniformMixedLayoutFromWidths(lattice,Uint8Array.from(mask,m=>m?1:4),snapped);
    }else{
      const forced=this.mixedSolidPromotion().forced;
      layout=createUniformMixedLayout(lattice,regions,1,forced);assertUniformMixedSolidPromotion(layout,forced);
    }
    const built=performance.now(),refusal=this.mixedFrame!.updateLayout(layout);
    if(refusal)return refusal;
    this.mixedGeneration++;this.mixedHostFine=layout.fineTiles.length;
    Object.assign(this.executionInfo,{uniformMixedGeneration:this.mixedGeneration,uniformMixedFineTiles:layout.fineTiles.length,
      uniformMixedCoarseTiles:layout.coarseTiles.length,
      uniformMixedOwners:layout.cellCount,uniformSimulationCellScale:undefined,
      uniformMixedLayoutBuild_ms:built-started,uniformMixedLayoutApply_ms:performance.now()-built});
    this.publishContact(layout.fineTiles.length);
    return undefined;
  }
  /** Requested with no solid-contact request: a static solid requests
   * nothing (its tiles are cut 4h owners). Bodies request through the census. */
  private mixedRequestOf():boolean{
    const lattice=refinementRegionLattice(this.scene);
    const set=buildUniformDetailRequests({settings:this.detail,regions:this.scene.fluid.refinementRegions??[],domain:uniformDetailDomain(lattice),acceptedStep:this.detailAcceptedStep});
    this.detailUnconstrained=set.unconstrainedRegions.length;
    // No automatic source outside Dynamic: its planner state goes with it.
    this.detailPlanner=undefined;this.detailPlan=undefined;
    return this.setMixedRequest({requests:set.requests,solid:this.detail.solidContact?this.mixedSolidPromotion():undefined});
  }
  /** Bring the kept request to `input` by what differs: the tiles that
   * entered and left the request boxes, and every tile's promotion bit only
   * under another solid mask or contact setting. True: a tile or the solid
   * mask moved, so the layout a build would give did. */
  private setMixedRequest(input:NonNullable<WebGPUUniformReferenceSolver["mixedRequest"]>["input"]):boolean{
    let request=this.mixedRequest;const lattice=refinementRegionLattice(this.scene),held=request?.tiles.lattice;
    // Another lattice names other tiles: its request starts empty.
    if(!request||!held||JSON.stringify(held)!==JSON.stringify(lattice)){const tiles=new UniformDetailRequestTiles(lattice);request=this.mixedRequest={tiles,mask:new Uint8Array(tiles.tiles),count:0,input:{requests:[]},moved:[]};}
    const {mask,tiles,moved}=request,{entered,left}=tiles.set(input.requests),forced=input.solid?.forced,solids=forced!==request.input.solid?.forced;
    if(solids){
      const fine=tiles.fine;let count=0;
      for(let t=0;t<mask.length;t++){const m=fine[t]!|(forced?forced[t]!<<1:0);mask[t]=m;if(m)count++;}
      request.count=count;
    }else{
      for(const t of entered){if(!mask[t])request.count++;mask[t]!|=1;}
      for(const t of left)if(!(mask[t]!&=2))request.count--;
    }
    if(this.mixedStaticsOf===tiles){for(const t of entered)moved.push(t);for(const t of left)moved.push(t);}
    request.coupled=input.solid?.any?input.solid.coupled:undefined;request.input=input;
    request.edit=solids?undefined:{tiles:entered.length+left.length,reach:this.mixedEditReach(tiles.tileDimensions,entered,left)};
    return solids||entered.length>0||left.length>0;
  }
  /** The layout's owner, between frames. One path: the GPU relayout (census,
   * builder, adopt in the frame head). Dynamic attaches it with the
   * importance criteria on. Requested and Full run the same relayout with
   * them off (uniformDetailImportance), so it builds from requests only: the
   * plan's tiles as builder statics, the solid-contact request as the
   * census's liquid-conditional promotion, bodies as the tiles their GPU
   * poses sweep. It is attached while the host's bound on those requests is
   * not zero. With no request nothing is attached: the all-4h host layout,
   * C = 0, base blocks only, no census. Capacity is one rule for every
   * policy (followMixedCapacity). hostMixedLayout says what the host still
   * builds. bodies: the roster the next frame advances (absent: the last
   * one's). advancing: the frame is next (Dynamic's statics are planned
   * once per accepted step). False: the device refused the request's h
   * tiles; the accepted generation keeps running and uniformDetail.rejected
   * names why. */
  private syncMixedLayout(bodies:readonly RigidBodyState[]=this.mixedRoster,dt=this.mixedRosterDt,advancing=false):boolean{
    const frame=this.mixedFrame;if(!frame)return true;
    const first=this.mixedGeneration===0,tiles=frame.ownership.capacity.tiles;
    this.mixedRoster=bodies;this.mixedRosterDt=dt;
    const taken=this.takeMixedReceipts();
    if(this.detail.policy==="dynamic"){
      // The census asks for any tile: the h store spans the lattice.
      this.windowMixed(true);
      if(!this.mixedGpuLayout||!this.mixedCensus){
        if(first){
          const refusal=this.hostMixedLayout();if(refusal)throw new Error(`Uniform simulation detail: ${refusal}`);
          this.publishDetail(this.planDetail());
        }
        // Every tile at the attach: frame 1's head runs on the all-h start,
        // and a census's first build may ask for any tile. The receipts
        // return what it does not use.
        frame.setRelayout(this.mixedRelayout());
        this.mixedGpuLayout=true;this.mixedCensus=true;this.mixedRequest=undefined;this.resetMixedCapacity();
        this.mixedBuilderStaticKey=undefined;this.mixedRegionKey="";this.mixedRefusedKey="";this.mixedLive=false;this.mixedSynced=undefined;this.mixedEditJoin=undefined;
      }
      // Static h tiles reach the builder here, ahead of the build that reads
      // them, so the capacity holds the new ones before any receipt counts them.
      if(advancing&&this.detailStaticKey()!==this.mixedBuilderStaticKey){
        const before=this.detailPlan?.fine;this.refreshMixedBuilderStatics(this.mixedSolidPromotion());
        const after=this.detailPlan!.fine;let added=0;
        for(let t=0;t<after.length;t++)if(after[t]&&!before?.[t])added++;
        if(added)this.mixedStaticsAdded={tiles:added+(this.mixedStaticsAdded?.tiles??0),build:this.mixedBuilds};
      }
      const swept=bodies.length?this.mixedBodyTiles(bodies,dt):undefined,joined=this.mixedEditJoin?.tiles;
      let extra=this.mixedStaticsAdded?.tiles??0;
      if(swept||joined)for(let t=0;t<tiles;t++)if(swept?.[t]||joined?.[t])extra++;
      // A census with nowhere to grow has no generation to fall back to.
      const refusal=this.followMixedCapacity(tiles,extra,true,true);
      if(refusal)throw new Error(`Uniform simulation detail: ${refusal}`);
      return true;
    }
    // Nothing asked since the last sync and no body to bound: only the capacity follows its receipts.
    const synced=this.mixedSynced;
    if(!first&&!bodies.length&&!this.mixedRosterKey&&synced?.scene===this.scene&&synced.detail===this.detail&&synced.stamp===this.mixedSolidMaskStamp
      &&!this.mixedEditJoin&&!taken.released){
      const request=this.mixedRequest;
      // The window follows the receipts too: a box no build holds any more leaves it.
      if(this.mixedGpuLayout&&request)this.windowMixed(false);
      if(this.mixedGpuLayout&&request&&taken.any){
        const refusal=this.followMixedCapacity(request.count,0,this.mixedLive,false);
        if(refusal){this.executionInfo.uniformDetail={...this.executionInfo.uniformDetail!,rejected:refusal};return false;}
      }
      return true;
    }
    const requestKey=`${JSON.stringify(this.scene.fluid.refinementRegions??[])}:${this.detail.policy}:${this.detail.solidContact}`,key=`${this.mixedSolidMaskStamp}:${requestKey}`;
    // The roster, not its poses: the census follows those on the GPU.
    const roster=bodies.length?JSON.stringify(bodies.map(b=>[b.description.id,b.description.shape,b.description.dimensions_m,b.description.motion])):"";
    // A generation Dynamic built: the requests take its relayout over where it stands.
    const census=this.mixedGpuLayout&&this.mixedCensus;
    let fresh=false,moved=false;
    // A candidate that is not accepted leaves the kept request as it was.
    const kept=this.mixedRequest,prior=kept&&{input:kept.input,moved:kept.moved.length};
    const keep=()=>{
      if(!fresh)return;
      if(this.mixedRequest!==kept)this.mixedRequest=kept;
      else if(prior){this.setMixedRequest(prior.input);kept!.moved.length=prior.moved;}
    };
    // Refused before: the same request is not asked again.
    if(!prior||(key!==this.mixedRegionKey&&key!==this.mixedRefusedKey)){moved=this.mixedRequestOf();fresh=true;}
    // A refused scene request was taken back: the layout answers the request again.
    else if(key===this.mixedRegionKey&&this.mixedRefusedKey){this.mixedRefusedKey="";this.executionInfo.uniformDetail={...this.executionInfo.uniformDetail!,rejected:undefined};}
    // Another request or owner: no receipt counts it, so the host's bound
    // holds the capacity again. A solid edit or a roster change alone keeps
    // the receipts: its tiles are the join's and the bodies' reach, counted below.
    const edited=fresh&&!census&&this.mixedRegionKey.slice(this.mixedRegionKey.indexOf(":")+1)===requestKey;
    if(fresh&&!edited)this.resetMixedCapacity();
    // A request whose tiles and solid mask stand (a region moved inside its tiles) keeps the layout it has.
    // One that moved request tiles only, on a layout nothing but the requests hold
    // (no solid contact, body or join: an empty census band), is a counted edit.
    const request=this.mixedRequest!;
    if((fresh&&(moved||census))||roster!==this.mixedRosterKey)
      this.staleMixed(fresh&&!census&&roster===this.mixedRosterKey&&this.mixedGpuLayout&&!this.mixedLive&&!bodies.length&&!request.coupled&&!this.mixedEditJoin?request.edit:undefined);
    this.mixedRosterKey=roster;
    // The bound: the request's tiles, the tiles the roster's poses can reach
    // this step and a live edit's join. extra: those the host adds in one
    // frame with no liquid moving, over whatever the receipts count.
    const mask=request.mask,swept=bodies.length&&request.count<tiles?this.mixedBodyTiles(bodies,dt):undefined,joined=this.mixedEditJoin?.tiles;
    let need=request.count,extra=0;
    if(swept||joined)for(let t=0;t<tiles;t++){if(!mask[t]&&(swept?.[t]||joined?.[t])){need++;extra++;}else if(joined?.[t])extra++;}
    need=Math.min(tiles,need);
    // The h store's window, before anything below reserves: zero detail holds no store.
    if(need>0)this.windowMixed(bodies.length>0,first);else{this.mixedWindows.length=0;this.mixedWindowOf=undefined;}
    // The liquid moves the need: solid contact over solids, bodies.
    const live=request.count<tiles&&(bodies.length>0||!!request.coupled);
    const refuse=(refusal:string)=>{
      // Nothing accepted yet: there is no generation to keep running.
      if(first)throw new Error(`Uniform simulation detail: ${refusal}`);
      keep();
      if(fresh)this.mixedRefusedKey=key;
      this.executionInfo.uniformDetail={...this.executionInfo.uniformDetail!,rejected:refusal};
      return false;
    };
    // Haloed patches (QA) hold their first layout: no relayout of either kind.
    const patch=!!this.detailStorage?.layout.patch;
    if(this.mixedGpuLayout&&need===0){
      // Nothing is requested: zero detail is the host's all-4h layout, C = 0.
      frame.setRelayout(undefined);this.mixedGpuLayout=false;this.mixedCensus=false;this.mixedReserved=undefined;this.mixedCap=undefined;
      const refusal=this.hostMixedLayout(mask);
      if(refusal)throw new Error(`Uniform simulation detail: ${refusal}`);
      this.publishMixedCapacity();
    }else if(!this.mixedGpuLayout&&(first||(fresh&&(need===0||patch)))){
      const refusal=this.hostMixedLayout(mask);if(refusal)return refuse(refusal);
      this.publishMixedCapacity();
    }
    if(need>0&&!patch){
      if(!this.mixedGpuLayout){
        // The accepted layout runs until the relayout's pipelines exist.
        const missing=this.pipelineNeeds.missing(["dynamic",...(this.detailStorage?.layout.domain?["transfer" as const]:[])]);
        if(missing.length){this.awaitPipelines(missing);this.mixedLive=false;keep();return true;}
      }
      // Admission before the request changes anything: the capacity its bound needs.
      let refusal=this.followMixedCapacity(need,extra,live,false);
      // Dynamic's generation stays; its band was the liquid's, a request's is every h tile.
      if(!refusal&&census){refusal=frame.reserveFine(frame.ownership.capacity.fineTiles,false,true);this.refreshMixedAllocation();}
      if(refusal)return refuse(refusal);
      if(!this.mixedGpuLayout){
        this.refreshRequestStatics(request);
        // On the storage the rule just reserved. The attach builds. From
        // all-4h, a request nothing but the host moves (no body, solid
        // contact or join) is a counted edit of that layout: its tiles and
        // their reach bound the build, and the frame keeps its budgets.
        let counted:UniformWorkEdit|undefined;
        if(!first&&this.mixedHostFine===0&&!bodies.length&&!request.coupled&&!this.mixedEditJoin){
          const listed:number[]=[];for(let t=0;t<mask.length;t++)if(mask[t])listed.push(t);
          counted={tiles:listed.length,reach:this.mixedEditReach(request.tiles.tileDimensions,listed)};
        }
        frame.setRelayout(this.mixedRelayout(),frame.ownership.capacity.fineTiles,!!counted);this.mixedGpuLayout=true;this.staleMixed();this.mixedEdit=counted;
      }else if(fresh||census)this.refreshRequestStatics(request);
      this.mixedCensus=false;this.publishMixedCapacity();
    }
    if(fresh){this.mixedRegionKey=key;this.mixedRefusedKey="";this.publishDetail({diagnostics:request.tiles.diagnostics(this.detail.budgetPercent)});}
    else if(!this.mixedRefusedKey&&this.executionInfo.uniformDetail?.rejected)this.executionInfo.uniformDetail={...this.executionInfo.uniformDetail,rejected:undefined};
    // Every tile already h (Full): nothing for a census to follow.
    this.mixedLive=this.mixedGpuLayout&&live;
    this.mixedSynced={scene:this.scene,detail:this.detail,stamp:this.mixedSolidMaskStamp};
    return true;
  }
  /** A new request or owner: its receipts start again, and a return staged for the last one is taken back. */
  private resetMixedCapacity():void{
    this.mixedRequestEpoch++;this.mixedNeeds=[];this.mixedQuiet=0;this.mixedHinted=true;this.mixedDeferredNeed=0;this.mixedCap=undefined;this.mixedStaticsAdded=undefined;
  }
  /** The receipts that arrived since the last sync, in the rule's window.
   * released: a live edit's join was admitted and its tiles follow the
   * requests again. A build under the staged cap that fit it confirms the cap. */
  private takeMixedReceipts():{any:boolean;released:boolean}{
    const rule=this.mixedCapacityRule,sync=++this.mixedSyncs;let any=false,released=false;
    while(this.mixedReceipts.length&&this.mixedReceipts[0]!.sync+rule.lagFrames<sync){
      const r=this.mixedReceipts.shift()!,over=r.need>r.admit,join=this.mixedEditJoin;any=true;
      // An admitted build was adopted: the boxes of the requests before the one it saw hold no h tile any more.
      if(!over){const queue=this.mixedWindows;let seen=-1;for(let i=0;i<queue.length;i++)if(queue[i]!.build<=r.build)seen=i;if(seen>0)queue.splice(0,seen);}
      if(join?.build!==undefined&&r.build>=join.build){
        if(over){
          // Deferred with the edit: its displacement ran on the tiles that were
          // already h. The join and the displacement are made again for the
          // build the grown capacity admits.
          join.build=undefined;this.mixedDynamic!.join(join.tiles);this.mixedFrame!.editSolids();
        }else{this.mixedEditJoin=undefined;this.staleMixed();released=true;}
      }
      if(this.mixedStaticsAdded&&r.build>=this.mixedStaticsAdded.build)this.mixedStaticsAdded=undefined;
      if(r.epoch!==this.mixedRequestEpoch)continue;
      this.mixedHinted=false;this.mixedQuiet++;this.mixedNeeds.push(r.need);if(this.mixedNeeds.length>rule.windowBuilds)this.mixedNeeds.shift();
      if(over){
        // The build kept its generation: the next sync holds its need and the head builds again.
        this.mixedDeferredNeed=Math.max(this.mixedDeferredNeed,r.need);this.staleMixed();this.mixedQuiet=0;
        this.mixedDeferrals.builds++;this.mixedDeferrals.tiles+=Math.max(0,r.need-r.fine);
      }
      const cap=this.mixedCap;
      if(cap&&r.build>=cap.build){if(over)this.mixedCap=undefined;else if(r.admit===cap.tiles)cap.confirmed=true;}
    }
    return {any,released};
  }
  /** One h-tile capacity rule for Requested, Dynamic and Full. The storage
   * holds C h tiles and the builder admits A <= C: a build that asks for
   * more keeps its generation and its receipt says what it needed.
   * bound: the host's count of what the next build may ask. It holds the
   * capacity until a receipt of this request arrives (the first-frame hint,
   * which covers solid contact), and for a request only the host changes it
   * is exact and stays the rule. extra: tiles the host adds in one frame
   * with no liquid moving (the bodies' reach, a live edit's join, static
   * tiles newly planned), held above what the receipts count. live: the
   * liquid moves the need (a census), so the capacity keeps the rule's
   * headroom over the last receipt (UniformDetailCapacityRule) and returns
   * after a window of them. Growth is between frames (reserveFine). A
   * return is staged: A is lowered, and the storage follows when a receipt
   * of a build under the cap confirms that no later generation holds more.
   * Returns the device's refusal of a growth, with nothing changed. */
  private followMixedCapacity(bound:number,extra:number,live:boolean,liquidBand:boolean):string|undefined{
    const frame=this.mixedFrame!,rule=this.mixedCapacityRule,T=frame.ownership.capacity.tiles,held=frame.ownership.capacity.fineTiles;
    if(frame.capacityFixed){
      // QA: a fixed capacity does not follow anything. A census over it is
      // the builder's fatal; a request over it has nowhere to run.
      const most=Math.max(bound,this.mixedDeferredNeed);
      if(!liquidBand&&most>held)throw new Error(`Uniform mixed ownership needs ${most} h tiles; its owner-indexed storage holds ${held}`);
      this.admitMixed(liquidBand?undefined:held);return undefined;
    }
    const last=Math.max(this.mixedNeeds.at(-1)??0,this.mixedDeferredNeed),peak=Math.max(last,...this.mixedNeeds);
    const hint=this.mixedHinted||rule.hold||!live?bound:0;
    const hold=(need:number,fraction:number)=>Math.min(T,Math.ceil((1+fraction)*need)+extra);
    // What must fit now, and what a growth reserves.
    const must=Math.max(hint,this.mixedDeferredNeed,live&&rule.grow>=0?hold(last,rule.grow):0);
    let cap=this.mixedCap;
    if(cap&&must>cap.tiles)cap=this.mixedCap=undefined;
    if(must>held){
      const want=Math.max(must,live?Math.min(T,hold(last,rule.headroom)+rule.floorTiles):0,rule.double&&held>0?Math.min(T,2*held):0);
      let reservation=frame.fineReservation(want,liquidBand,live);
      if(reservation.refusal&&want>must)reservation=frame.fineReservation(must,liquidBand,true);
      if(reservation.refusal)return reservation.refusal;
      const refusal=frame.reserveFine(reservation.fineTiles,liquidBand,true);if(refusal)return refusal;
      this.mixedQuiet=0;this.mixedDeferrals.changes++;
    }else if(cap?.confirmed){
      // Every generation since the cap fits it: the storage returns.
      const refusal=frame.reserveFine(cap.tiles,liquidBand,true);if(refusal)return refusal;
      this.mixedCap=undefined;this.mixedDeferrals.changes++;
    }else if(!cap&&!rule.hold){
      // A census returns to its window's peak, held as a growth would hold
      // it; a host request to what a host layout of it would reserve.
      const want=live?Math.min(T,hold(peak,rule.headroom)+rule.floorTiles):frame.fineReservation(bound,liquidBand).fineTiles;
      // The cap holds what the requests ask: the build under it moves no tile (a zero edit; a census counts none).
      if(live?!this.mixedHinted&&this.mixedQuiet>=rule.windowBuilds&&want*rule.returnRatio<=held:want<held){this.mixedCap={tiles:want,build:this.mixedBuilds};this.staleMixed(live?undefined:{tiles:0,reach:0});}
    }
    if(this.mixedDeferredNeed<=frame.ownership.capacity.fineTiles)this.mixedDeferredNeed=0;
    const admission=this.mixedCap?.tiles??frame.ownership.capacity.fineTiles;
    this.admitMixed(admission,!live&&!liquidBand&&bound<=admission);
    return undefined;
  }
  /** The builder's admission capacity. A capacity of every tile admits anything: no admit launch.
   * fits: it holds everything a host-counted request asks, so the build under it moves no tile (a zero edit). */
  private admitMixed(tiles?:number,fits=false):void{
    const frame=this.mixedFrame!,capacity=frame.ownership.capacity;
    if(tiles!==this.mixedReserved){this.mixedBuilder!.setAdmission(tiles===undefined||tiles>=capacity.tiles?undefined:tiles);this.mixedReserved=tiles;this.staleMixed(fits?{tiles:0,reach:0}:undefined);}
    this.publishMixedCapacity();
  }
  private publishMixedCapacity():void{
    const capacity=this.mixedFrame!.ownership.capacity.fineTiles,d=this.mixedDeferrals;
    if(capacity!==this.executionInfo.uniformMixedFineCapacity)this.refreshMixedAllocation();
    Object.assign(this.executionInfo,{uniformMixedFineCapacity:capacity,uniformMixedAdmission:this.mixedGpuLayout?this.mixedReserved??capacity:capacity,
      uniformMixedDeferredBuilds:d.builds,uniformMixedDeferredTiles:d.tiles,uniformMixedCapacityChanges:d.changes});
  }
  /** Plan detail for the current policy, regions and latest focus at the
   * accepted step, and commit it (the planner's automatic state advances). */
  private planDetail():UniformDetailPlan{
    const lattice=refinementRegionLattice(this.scene),planner=this.detailPlanner??=new UniformDetailPlanner(lattice);
    const focus=this.detailFocusPending,step=this.detailAcceptedStep;
    const set=buildUniformDetailRequests({settings:this.detail,regions:this.scene.fluid.refinementRegions??[],domain:uniformDetailDomain(lattice),acceptedStep:step,focus});
    const plan=planner.plan({requests:set.requests,suppressions:set.suppressions,automaticSources:set.automaticSources,acceptedStep:step,budgetPercent:this.detail.budgetPercent});
    planner.commit(plan);
    const suppressed=new Uint8Array(planner.tiles);
    for(const box of set.suppressions)for(const t of planner.tilesOf(box.bounds_m))suppressed[t]=1;
    this.detailPlan=plan;this.detailSuppressed=suppressed;this.detailFocus=focus;this.detailUnconstrained=set.unconstrainedRegions.length;
    return plan;
  }
  /** The accepted plan's diagnostics, beside what this solver adds outside it. */
  private publishDetail(plan:Pick<UniformDetailPlan,"diagnostics">):void{
    const d=plan.diagnostics,policy=this.detail.policy,contactTiles=this.contactTilesOf(this.executionInfo.uniformMixedFineTiles??0,d.admittedTiles);
    this.executionInfo.uniformDetail={policy,requestKey:uniformDetailRequestKey(policy,this.scene.fluid.refinementRegions),
      focusRevision:policy==="dynamic"&&this.detail.nearFocus?this.detailFocus?.revision:undefined,
      tiles:d.tiles,requestedTiles:d.requestedTiles,admittedTiles:d.admittedTiles,supportTiles:d.supportTiles,deferredTiles:d.deferredTiles,
      budgetClippedTiles:d.budgetClippedTiles,budgetTiles:d.budgetTiles,automaticCostTiles:d.automaticCostTiles,contactTiles,
      patchCells:d.patchCells,residentPatches:d.residentPatches,allocatedCells:d.allocatedCells,activeCells:d.activeCells,
      supportCells:d.supportCells,wastedCells:d.wastedCells,reasons:d.reasons,unconstrainedRegions:this.detailUnconstrained,
      mapping:policy==="full"?"Every tile at h (GPU layout, requests only)."
        :policy==="requested"?uniformDetailCoarseSolids(this.detail)?"4h background, static solids on cut 4h owners; Fine regions at h, bodies at h where they meet liquid (GPU layout, requests only)."
        :"4h background; Fine regions at h, solids and bodies at h where they meet liquid (GPU layout, requests only)."
        :"Interim: the GPU surface census picks h or 4h each frame; Fine regions and focus tiles are held h, Coarse regions 4h except solid contact.",
      preparing:this.pipelinePreparing};
  }
  /** h tiles beyond the plan's under Requested and Full: solid and body
   * contact (for one build, a live edit's tiles). Dynamic's census picks
   * its own tiles, so it reports none. */
  private contactTilesOf(fine:number,admitted:number):number{return this.detail.policy==="dynamic"?0:Math.max(0,fine-admitted);}
  /** The running layout's h tiles changed (a host layout, a build's receipt). */
  private publishContact(fine:number):void{
    const d=this.executionInfo.uniformDetail;if(!d)return;
    const contactTiles=this.contactTilesOf(fine,d.admittedTiles);
    if(contactTiles!==d.contactTiles)this.executionInfo.uniformDetail={...d,contactTiles};
  }
  /** Transient focus: the latest revision wins, consumed at the next frame head. */
  applyDetailInput(input:SolverDetailInput):void{
    const focus=input.focus;
    if(focus&&(!this.detailFocusPending||focus.revision!==this.detailFocusPending.revision))this.detailFocusPending=focus;
  }
  /** Dynamic statics are current while this key holds: replanned on a focus
   * revision, a setting, or each accepted step while hysteresis or churn is pending. */
  private detailStaticKey():string{
    const pending=this.detailPlanner&&(!this.detailPlanner.settled||(this.detailPlan?.diagnostics.churnDeferredTiles??0)>0);
    const focus=this.detail.nearFocus?this.detailFocusPending?.revision??0:0;
    return `${this.mixedSolidMaskStamp}:${JSON.stringify(this.scene.fluid.refinementRegions??[])}:${JSON.stringify(this.detail)}:${focus}${pending?`:${this.detailAcceptedStep}`:""}`;
  }
  /** Horizon one: the frame head classifies the state the frame starts from
   * for this frame's dt and advects on the generation it builds. Smooth
   * surface tiles may use 4h; detail and impact protection retain h. This
   * also selects surface pressure resolution, so broad motion and sheets
   * are assessed by properties rather than an identical fine trajectory. */
  private mixedRelayout():UniformMixedFrameRelayout{
    const dynamic=this.mixedDynamic!,builder=this.mixedBuilder!;
    return {generation:builder.generation,receipt:builder.receipt,reasons:dynamic.reasons,importance:dynamic.importance,
    // Requests only: a layout only the host changes is built once per change, and the frames between run the head without the census.
    due:()=>this.detail.policy==="dynamic"||this.mixedLive||this.mixedStale,
    edit:()=>this.detail.policy==="dynamic"||this.mixedLive||!this.mixedStale?undefined:this.mixedEdit,encode:(encoder,dt,views)=>{
      // Requests-only statics are the host's, refreshed with their admission (syncMixedLayout).
      if(this.detail.policy==="dynamic"&&this.detailStaticKey()!==this.mixedBuilderStaticKey)this.refreshMixedBuilderStatics(this.mixedSolidPromotion());
      // One more build after the last live frame returns what its bodies held.
      this.mixedStale=this.mixedLive;this.mixedEdit=undefined;
      const build=this.mixedBuilds++,join=this.mixedEditJoin;if(join&&join.build===undefined)join.build=build;
      const g=this.scene.fluid.gravity_m_s2;
      // The detail controls' importance criteria: what the census requires at h.
      const importance=uniformDetailImportance(this.detail,dynamic.ownership.capacity.tiles);
      // The build's counters clear in the census's blit run.
      builder.encodeClear(encoder);
      if(this.detail.policy==="dynamic"&&Object.values(importance.criteria).some(Boolean))this.mixedFrame?.narrowBandFlip?.refine(encoder,dynamic.joinTarget(),dt);
      dynamic.encode(encoder,{dt,steps:1,gravity:[g.x,g.y,g.z],reach:Math.max(this.mixedCoarseningReach,this.detail.marginTiles),hysteresis:this.mixedCoarseningHysteresis,
        surfaceTolerance:importance.shapeTolerance,fastTravel:0,importance,
        boundaryTravel:importance.impactTravel,closedWalls:this.scene.container.top==="open"?0b101111:0b111111,up:Math.sign(-g.y),
        fullTolerance:UNIFORM_MIXED_DYNAMIC_FULL_TOLERANCE,emptyTolerance:Math.max(this.volumeDustThreshold,1e-6),reasons:views},false);
      builder.encode(encoder);
    },adopted:encoder=>{
      // Diagnostics only: a free read takes this build's receipt; mapped
      // once the frame submits (readMixedRelayout), never awaited.
      const read=this.mixedRelayoutReads.pop();
      if(read){const r=builder.receipt;encoder.copyBufferToBuffer(r.buffer,r.offset,read,0,r.words*4);this.mixedRelayoutRead=read;}
    }};
  }
  /** Map the relayout receipt the frame just submitted into executionInfo. */
  private readMixedRelayout():void{
    const read=this.mixedRelayoutRead;this.mixedRelayoutRead=undefined;if(!read)return;
    const sequence=++this.mixedRelayoutSequence,epoch=this.mixedRequestEpoch,host=this.mixedGeneration,build=this.mixedBuilds-1,sync=this.mixedSyncs;
    read.mapAsync(GPUMapMode.READ).then(()=>{
      const r=new Uint32Array(read.getMappedRange()).slice();read.unmap();this.mixedRelayoutReads.push(read);
      const R=UNIFORM_MIXED_RELAYOUT_RECEIPT,C=R.census,f=new Float32Array(r.buffer);
      if(this.disposed)return;
      // The h tiles the build asked for, the admission it ran under and the
      // generation it left: the next sync's capacity (takeMixedReceipts).
      this.mixedReceipts.push({build,epoch,sync,need:r[R.need]!,admit:r[R.admit]!,fine:r[R.tiers]!});
      if(sequence<this.mixedRelayoutShown)return;
      this.mixedRelayoutShown=sequence;
      // A fatal build latches frame failure 6/7 on the GPU (the remap); the
      // frame's receipt names it. Nothing here decides anything.
      // A host layout replaced the generation this build belongs to: its counts are not the running layout's.
      if(host!==this.mixedGeneration||!this.mixedGpuLayout)return;
      const generation=r[R.generation]!,fine=r[R.tiers]!,coarse=r[R.tiers+1]!;
      this.publishContact(fine);
      Object.assign(this.executionInfo,{uniformMixedGeneration:this.mixedGeneration+generation,
        uniformMixedFineTiles:fine,uniformMixedCoarseTiles:coarse,uniformMixedOwners:64*fine+coarse,uniformSimulationCellScale:undefined,
        uniformMixedLayoutBuild_ms:0,uniformMixedLayoutApply_ms:0,
        uniformMixedDynamicRelayouts:generation,uniformMixedDynamicChangedTiles:r[R.changed]!,
        uniformMixedDynamicInterfaceTiles:r[C]!,uniformMixedDynamicBandTiles:r[C+2]!,
        uniformMixedDynamicRefined:r[C+4]!,uniformMixedDynamicCoarsened:r[C+5]!,uniformMixedDynamicRequiredTiles:r[C+12]!,uniformMixedDynamicBoundaryTiles:r[C+16]!,uniformMixedDynamicSolidTiles:r[C+19]!,uniformMixedDynamicUnresolvedCoarse:r[C+1]!,
        uniformMixedDynamicCoarsePartialVolume:r[C+6]!,uniformMixedDynamicCoarsePhiCrossing:r[C+7]!,uniformMixedDynamicCoarseDryLiquidPhi:r[C+3]!,
        uniformMixedDynamicInteriorDeficit:[f[C+8]!,f[C+9]!],uniformMixedDynamicAirVolume:[f[C+10]!,f[C+11]!]});
    },error=>{if(!this.disposed)this.failMixedFrame(error);});
  }
  /** Tiles a roster's bodies can touch over `dt` (bounding sphere dilated by
   * its travel and one cell), plus one tile around them: the host's count of
   * what the census promotes around the GPU poses (UniformMixedBodies
   * .encodeTiles via setBodies, liquid-conditional). The roster's poses and
   * speeds may be older than the GPU's: a census that needs more defers its
   * build and its receipt grows the capacity (readMixedRelayout). */
  private mixedBodyScratch?:Uint8Array;
  private mixedBodyTiles(bodies:readonly RigidBodyState[],dt:number):Uint8Array{
    const {dimensions,cellSize_m,origin_m}=refinementRegionLattice(this.scene);
    const n=dimensions.map(d=>d/4),count=n[0]!*n[1]!*n[2]!,cell=Math.max(...cellSize_m);
    // One scratch mask per solver: the callers count it and keep nothing.
    const tiles=this.mixedBodyScratch?.length===count?this.mixedBodyScratch.fill(0):(this.mixedBodyScratch=new Uint8Array(count));
    for(const body of bodies){
      const v=body.linearVelocity_m_s,reach=boundingRadius(body)+Math.hypot(v.x,v.y,v.z)*dt+cell;
      const centre=[body.position_m.x-origin_m.x,body.position_m.y-origin_m.y,body.position_m.z-origin_m.z];
      const tile=(axis:number,offset:number)=>Math.min(n[axis]!-1,Math.max(0,Math.floor((centre[axis]!+offset)/(4*cellSize_m[axis]!))+Math.sign(offset)));
      const lo=[0,1,2].map(a=>tile(a,-reach)),hi=[0,1,2].map(a=>tile(a,reach));
      for(let z=lo[2]!;z<=hi[2]!;z++)for(let y=lo[1]!;y<=hi[1]!;y++)for(let x=lo[0]!;x<=hi[0]!;x++)tiles[x+n[0]!*(y+n[1]!*z)]=1;
    }
    return tiles;
  }
  /** The solid mask as the mixed frame last saw it, to find the tiles a live edit touched. */
  private mixedSolidWords?:Uint32Array;
  /** A live voxel edit on the mixed path. The frame's solid kernels read the
   * host mask directly; here every tile within one cell of a changed voxel
   * (and the solid-coupled tiles around it, liquid-conditional promotion's
   * rule) joins the next census's band (a GPU join), which the frame head
   * adopts before it displaces liquid out of the new solid. A requests-only
   * layout takes the edit through its request instead (the caller's
   * syncMixedLayout): the solid mask of the next census, when contact is
   * requested, and cut 4h owners otherwise. */
  private editMixedSolids(dirty:{firstWord:number;wordCount:number}):void{
    const frame=this.mixedFrame!,words=this.solidMask.words,previous=this.mixedSolidWords!;
    const {dimensions}=refinementRegionLattice(this.scene);
    const [nx,ny,nz]=dimensions,n=dimensions.map(d=>d/4) as [number,number,number];
    const sx=nx+2,sy=ny+2,touched=new Uint8Array(n[0]*n[1]*n[2]);
    const header=SOLID_OCCUPANCY_MASK_HEADER_WORDS;
    for(let w=Math.max(header,dirty.firstWord);w<dirty.firstWord+dirty.wordCount;w++){
      let bits=(words[w]!^previous[w]!)>>>0;
      while(bits){
        const bit=31-Math.clz32(bits&-bits);bits=(bits&(bits-1))>>>0;
        const index=(w-header)*32+bit,x=index%sx-1,y=Math.floor(index/sx)%sy-1,z=Math.floor(index/(sx*sy))-1;
        for(let tz=Math.max(0,(z-1)>>2);tz<=Math.min(n[2]-1,(z+1)>>2);tz++)
          for(let ty=Math.max(0,(y-1)>>2);ty<=Math.min(n[1]-1,(y+1)>>2);ty++)
            for(let tx=Math.max(0,(x-1)>>2);tx<=Math.min(n[0]-1,(x+1)>>2);tx++)touched[tx+n[0]*(ty+n[1]*tz)]=1;
      }
    }
    previous.set(words.subarray(dirty.firstWord,dirty.firstWord+dirty.wordCount),dirty.firstWord);
    if(this.mixedGpuLayout&&this.detail.policy==="dynamic"){
      const solid=this.mixedSolidPromotion();
      // Refresh the builder statics and census solid mask now: the next census reads them.
      this.refreshMixedBuilderStatics(solid);
      const promoted=uniformMixedLiquidSolidPromotion(n,solid.coupled,touched),held=this.mixedEditJoin;
      const join=touched.map((b,t)=>b|promoted[t]!|(held?held.tiles[t]!:0));
      this.mixedDynamic!.join(join);this.mixedEditJoin={tiles:join};
    }else if(this.detail.policy!=="dynamic"&&!this.detailStorage?.layout.patch){
      // Requests only: nothing else asks for a new solid's tiles when solid
      // contact is off, and a dry promotion skips them when it is on. The
      // join holds them at h for the build that displaces (syncMixedLayout
      // counts them into the capacity before it).
      const promoted=uniformMixedLiquidSolidPromotion(n,this.mixedSolidPromotion().coupled,touched),held=this.mixedEditJoin;
      const join=touched.map((b,t)=>b|promoted[t]!|(held?held.tiles[t]!:0));
      this.mixedDynamic!.join(join);this.mixedEditJoin={tiles:join};
    }
    frame.editSolids();
  }
  /** Static h tiles: the detail plan's admitted tiles (Fine regions, focus).
   * Solid promotion is liquid-conditional, in the census band
   * (solidActive/solidPromote). Static 4h tiles: Coarse regions outside the
   * plan, which mask the census band except where a solid could be promoted. */
  private refreshMixedBuilderStatics(solid:ReturnType<WebGPUUniformReferenceSolver["mixedSolidPromotion"]>):void{
    const plan=this.planDetail(),suppressed=this.detailSuppressed!;
    this.mixedBuilder!.setStatic(plan.fine,Uint8Array.from(suppressed,(s,t)=>s&&!plan.fine[t]&&!solid.forced[t]?1:0));
    this.mixedDynamic!.setStatic(plan.fine);this.mixedDynamic!.setSolid(solid.coupled);
    this.mixedStaticsOf=undefined;this.mixedSolidSet=solid.coupled;
    this.mixedBuilderStaticKey=this.detailStaticKey();
    this.publishDetail(plan);
  }
  /** A requests-only relayout's statics: the request's h tiles, no 4h mask (no
   * automatic source to mask), and the census's solid mask only when solid
   * contact is requested. Masks that already mirror this request are moved
   * by its tiles (the words holding them); a solid mask the census holds is
   * not written again. */
  private refreshRequestStatics(request:NonNullable<WebGPUUniformReferenceSolver["mixedRequest"]>):void{
    const builder=this.mixedBuilder!,dynamic=this.mixedDynamic!,fine=request.tiles.fine;
    if(this.mixedStaticsOf===request.tiles){builder.editStatic(request.moved,fine);dynamic.editStatic(request.moved,fine);}
    else{builder.setStatic(fine);dynamic.setStatic(fine);this.mixedStaticsOf=request.tiles;}
    request.moved.length=0;
    const coupled=request.coupled??(this.mixedNoSolid?.length===fine.length?this.mixedNoSolid:this.mixedNoSolid=new Uint8Array(fine.length));
    if(coupled!==this.mixedSolidSet){dynamic.setSolid(coupled);this.mixedSolidSet=coupled;}
    this.mixedBuilderStaticKey=this.detailStaticKey();
  }
  /** The pipelines a state can dispatch, from the dispatch sites: the solid
   * twins (select), bodies (UniformMixedBodies, the bodies-only record),
   * displacement (a live edit, a body), the relayout's census, builder and
   * changed-tile launches (Dynamic, or any request for an h tile:
   * uniformDetailRequests), the force set of the surface width
   * (coarseSurfaceTravel) with its capillary caches, and under the domain
   * placement the transfers of a capacity that may cross zero. Absent
   * fields are the accepted state's. */
  private pipelineNeedsOf(state:{solids?:boolean;bodies?:number;detail?:UniformDetailSettings;surfaceTension?:number;edit?:boolean;relayout?:boolean;regions?:SceneDescription["fluid"]["refinementRegions"]}):UniformPipelineNeed[]{
    const solids=state.solids??(!this.solidVoxelsEmpty||sceneHasTerrain(this.scene)),bodies=state.bodies??this.stepBodyCount;
    const detail=state.detail??this.detail,cached=uniformDetailCoarseSurface(detail);
    const needs:UniformPipelineNeed[]=[cached?"forceCache":"forceInline"];
    if(solids||bodies)needs.push("solids");
    if(bodies)needs.push("bodies","displace");
    if(state.edit)needs.push("displace");
    const requests=uniformDetailRequests(detail,{fineRegions:(state.regions??this.scene.fluid.refinementRegions??[]).some(r=>uniformRegionDetailTier(r)==="fine"),solids,bodies});
    // A live edit's tiles are joined to a census under every policy (editMixedSolids).
    const join=!!state.edit&&!this.detailStorage?.layout.patch;
    if(requests||join)needs.push("dynamic");
    if(cached&&(state.surfaceTension??this.scene.fluid.surfaceTension_N_m)>0)needs.push("capillary");
    if(this.detailStorage?.layout.domain&&(requests||state.relayout||state.edit||bodies))needs.push("transfer");
    return needs;
  }
  /** The wait a change is in, for the detail diagnostics. */
  private get pipelinePreparing():string|undefined{return this.pipelineWaiting.size?[...this.pipelineWaiting].join(", "):undefined;}
  private publishPipelinePreparing():void{
    const detail=this.executionInfo.uniformDetail;
    if(detail&&detail.preparing!==this.pipelinePreparing)this.executionInfo.uniformDetail={...detail,preparing:this.pipelinePreparing};
  }
  /** Whether a state must wait: its unbuilt pipelines are prepared, the
   * caller keeps the accepted state and holds the change for the replay. */
  private awaitPipelines(needs:UniformPipelineNeed[]):boolean{
    const missing=this.pipelineNeeds.missing(needs);if(!missing.length)return false;
    const fresh=missing.filter(need=>!this.pipelineWaiting.has(need));
    if(!fresh.length)return true;
    for(const need of fresh)this.pipelineWaiting.add(need);
    const flight:Promise<void>=this.pipelineNeeds.prepare(fresh).then(()=>{
      for(const need of fresh)this.pipelineWaiting.delete(need);
      if(this.pipelineFlight===flight)this.pipelineFlight=undefined;
      if(this.disposed)return;
      this.publishPipelinePreparing();
      if(this.pipelineWaiting.size)return;
      // A newer change deferred behind a frame supersedes the one that waited.
      const scene=this.pendingPipelineScene,values=this.pendingPipelineValues;
      this.pendingPipelineScene=undefined;this.pendingPipelineValues=undefined;
      if(scene&&!this.deferredFrameScene)this.applySceneUniforms(scene);
      if(values&&!this.deferredFrameValues)this.applyRuntimeValues(values);
    }).catch(error=>{if(this.pipelineFlight===flight)this.pipelineFlight=undefined;this.failMixedFrame(error);});
    this.pipelineFlight=flight;this.publishPipelinePreparing();
    return true;
  }
  /** What a scene would change of the pipeline state: its solids against the accepted mask. */
  private pipelineSceneNeeds(scene:SceneDescription):UniformPipelineNeed[]{
    // The world the accepted mask was made from gives that mask: nothing to probe.
    const world=solidWorldForScene(scene);let edit=false,empty=this.solidVoxelsEmpty;
    if(!this.solidMask.describes(world)){
      const probe=new SolidOccupancyMask([this.executionInfo.nx,this.executionInfo.ny,this.executionInfo.nz]);probe.update(world);
      const words=this.solidMask.words;edit=probe.words.some((word,i)=>word!==words[i]);empty=uniformSolidMaskEmpty(probe);
    }
    const policy=this.detail.policy,relayout=uniformDetailRequestKey(policy,scene.fluid.refinementRegions)!==uniformDetailRequestKey(policy,this.scene.fluid.refinementRegions);
    return this.pipelineNeedsOf({solids:!empty||sceneHasTerrain(scene),surfaceTension:scene.fluid.surfaceTension_N_m,edit,relayout,regions:scene.fluid.refinementRegions??[]});
  }
  /** A live solid edit's pipelines, before the host adopts its scene
   * (applySceneUniforms then applies it at once). Nothing is committed here. */
  async prepareLiveSolidEdit(scene:SceneDescription):Promise<boolean>{
    if(this.mixedFrame&&this.pipelineNeeds.deferred&&this.awaitPipelines(this.pipelineSceneNeeds(scene)))await this.pipelinesPrepared();
    return false;
  }
  /** The host presents while a change waits (uniformDetail.preparing): the
   * accepted state keeps advancing. Without it advanceTo refuses to. */
  acceptPipelineWaits():void{this.pipelineWaitsAccepted=true;}
  /** Resolves once no change waits for pipelines and the replayed change has
   * been applied (a headless caller awaits it before stepping the new state). */
  async pipelinesPrepared():Promise<void>{
    while(this.pipelineFlight){await this.pipelineFlight;await this.mixedFrameChain;}
    if(this.pressureFrameFailure)throw this.pressureFrameFailure;
  }
  /** Builds every pipeline no accepted state has needed yet, pace at a time,
   * so a later edit, body or policy change rarely waits. The host calls it
   * once its first frame has presented; a run that never calls it compiles
   * nothing its states do not need. */
  warmPipelines(pace=UNIFORM_PIPELINE_WARM_PACE):Promise<void>{
    return this.pipelineNeeds.warm(pace).catch(error=>{this.failMixedFrame(error);});
  }
  /** Pipelines declared and not yet built (diagnostic). */
  get deferredPipelines():number{return this.pipelineNeeds.deferred;}
  /** A frame's encode or receipt failed: fatal for the solver. */
  private failMixedFrame(error:unknown):void{
    if(this.disposed||this.pressureFrameFailure)return;
    this.pressureFrameFailure=error instanceof Error?error:new Error(String(error));
    this.executionInfo.simulationPipelineError=this.pressureFrameFailure.message;
  }
  private assertMixedOptions(): void {
    assertUniformMixedOptions({
      velocityTransport: this.velocityTransport,
      liquidOnlyVelocityAdvection: this.liquidOnlyVelocityAdvection,
    });
  }

  private initializationTasks(signal?: AbortSignal): GPUInitializationTask[] {
    const tasks = [...this.rigidSystem.initializationTasks()];
    const compiler = gpuCompilationManagerFor(this.device);
    // Packed detail storage: the two native t=0 kernels address the fields
    // through the storage's accessors, their directory at group 1.
    const packed = this.detailStorage?.layout.placement === "packed" ? this.detailStorage : undefined;
    const shaderModule = compiler.createShaderModule({ label: "Uniform reference kernels",
      code: packed ? uniformDetailDenseShader(this.shaderSource, packed, UNIFORM_REFERENCE_DETAIL_CLASSES, 1) : this.shaderSource });
    const layout = packed ? this.device.createPipelineLayout({ bindGroupLayouts: [this.mainLayout, packed.dense.layout] }) : this.mainPipelineLayout;
    const compiled: Partial<UniformReferencePipelines> = {};
    // Uniform Geometric advances only through the mixed frame: of the native
    // kernels it dispatches just the t=0 dense authority seed (plus uvPublish).
    const programs = this.geometricVolume ? PIPELINES.filter(([key]) => key === "extrapolationAuthorityDense") : PIPELINES;
    const ids = programs.map(([key]) => `uniform.pipeline.${key}`);
    programs.forEach(([key, label, entryPoint], index) => tasks.push({
      id: ids[index], phase: "solver-pipelines", label,
      run: async () => {
        compiled[key] = await compiler.compileComputePipeline({
          label: `Uniform reference - ${entryPoint}`,
          layout,
          compute: { module: shaderModule, entryPoint },
        }, { priority: "visible", signal });
      },
    }));
    // The t=0 presentation is the only native geometric pass left: every
    // advance, and its publication, belongs to the mixed-ownership frame.
    if (this.geometricVolume) {
      const id = "uniform.volume.uvPublish"; ids.push(id);
      tasks.push({ id, phase: "solver-pipelines", label: "uvPublish", run: async () => {
        this.initialPublishPipeline = await compiler.compileComputePipeline({
          label: "Uniform Geometric - uvPublish", layout,
          compute: { module: shaderModule, entryPoint: "uvPublish" },
        }, { priority: "visible", signal });
      } });
    }
    if(this.fieldPages) { const id="uniform.fields.pages"; ids.push(id);
      tasks.push({id,phase:"solver-pipelines",label:"Uniform page publication",run:()=>this.fieldPages!.initialize(signal)}); }
    const pipelineReadyId = "uniform.pipeline.publish";
    tasks.push({ id: pipelineReadyId, phase: "solver-pipelines", label: "Publish uniform reference programs", dependencies: ids, run: () => {
      this.pipelines = !this.geometricVolume ? compiled as UniformReferencePipelines
        : new Proxy(compiled as UniformReferencePipelines, { get: (target, key) => {
          const pipeline = target[key as keyof UniformReferencePipelines];
          if (!pipeline) throw new Error(`Uniform Geometric does not compile the native ${String(key)} kernel`);
          return pipeline;
        } });
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
          shaderSource: `${this.shaderSource}\n${this.geometricVolume
            ? mixedSurfaceThetaFragment(this.pressureMultigrid.shaderFragment)
            : this.pressureMultigrid.shaderFragment}`,
          // The mixed frame enters the hierarchy at native 4h only.
          ...(this.geometricVolume ? { entryPoints: UNIFORM_MIXED_CONTINUATION_ENTRIES } : {}),
          signal,
        });
        this.publishUniformPipelineFacts();
      },
    });
    tasks.push({ id: "uniform.surface.initial", phase: "upload", label: "Reconstruct smooth t=0 surface", dependencies: [pipelineReadyId, extrapolationReadyId, multigridReadyId], run: () => { this.executionInfo.allocatedBytes += this.fieldPages?.allocationOverheadBytes ?? 0; if (!packed) this.encodeInitialPresentationSurface(); } });
    tasks.push({ id: "uniform.warmup", phase: "warmup", label: "Fence uniform t=0 uploads", dependencies: ["uniform.surface.initial"], run: async () => { await this.device.queue.onSubmittedWorkDone(); } });
    return tasks;
  }

  private encodeInitialPresentationSurface(): void {
    if (!this.pipelines) throw new Error("Uniform reference pipelines are not initialized");
    this.writeParams(0, Math.min(this.scene.rigidBodies.length, 12), 0);
    const encoder = this.device.createCommandEncoder({ label: "Uniform reference t=0 surface reconstruction" });
    // Seed static rho'/face-open authority once across the lattice. Runtime
    // updates only need the rolling active box; untouched air keeps this exact
    // static boundary state instead of paying the same full pass every frame.
    // Packed detail storage: the same two kernels through the storage's
    // accessors (initializationTasks), which deposit the canonical texels.
    const packed = this.detailStorage?.layout.placement === "packed" ? this.detailStorage : undefined;
    const groups = packed ? this.initialDetailGroups() : { authority: this.extrapolationAuthorityGroup, publish: this.wallFilmResolveGroup };
    const directory = packed?.dense.group;
    this.runDirect(encoder, "Uniform initial Sec. 3.3 interface authority",
      this.pipelines.extrapolationAuthorityDense, groups.authority,
      [Math.ceil(this.executionInfo.nx / 4), Math.ceil(this.executionInfo.ny / 4), Math.ceil(this.executionInfo.nz / 4)], directory);
    if (this.geometricVolume) {
      // Use the same page traversal as runtime publication. A dense-shaped
      // launch cannot cover a page-shaped entry point on a large domain.
      this.run(encoder, "Uniform Geometric initial page surface publication",
        this.initialPublishPipeline!, groups.publish, directory);
    } else if (this.densityPostProcessing) {
      this.run(encoder, "Uniform initial post-process blur x", this.pipelines.postprocessBlurX, this.postprocessBlurXGroup);
      this.run(encoder, "Uniform initial post-process blur y", this.pipelines.postprocessBlurY, this.postprocessBlurYGroup);
      this.run(encoder, "Uniform initial post-process blur z", this.pipelines.postprocessBlurZ, this.postprocessBlurZGroup);
      this.run(encoder, "Uniform initial sub-grid surface resolve", this.pipelines.postprocessResolve, this.postprocessResolveGroup);
    } else {
      this.run(encoder, "Uniform initial wall-film resolve", this.pipelines.wallFilmResolve, this.wallFilmResolveGroup);
    }
    // The two kernels store through no twin: beside an h generation the
    // fields' bases take their canonical texels from it.
    packed?.encodeBaseRefresh(encoder, [this.velocityD, this.velocityB, this.surfaceA, this.surfaceB, this.gammaB]);
    this.fieldPages?.encodePublications(encoder);
    this.device.queue.submit([encoder.finish()]);
  }

  /** The parameter block has been written: the all-4h solid record may be built from it. */
  private paramsWritten = false;
  private writeParams(dt: number, activeBodyCount: number, inflowStrength: number, drop?: InjectedLiquidBall): void {
    const c = this.scene.container;
    const inflow = this.scene.fluid.inflow;
    const outlet = this.inflowBoundary?.outletCenter_m;
    this.paramsWritten = true;
    this.device.queue.writeBuffer(this.params, 0, new Float32Array([
      this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz, dt,
      c.width_m / this.executionInfo.nx, c.height_m / this.executionInfo.ny, c.depth_m / this.executionInfo.nz, this.scene.fluid.gravity_m_s2.y,
      c.width_m, c.height_m, c.depth_m, sceneHasTerrain(this.scene) ? 1 : 0,
      this.scene.fluid.density_kg_m3, this.scene.fluid.dynamicViscosity_Pa_s,
      // The mixed frame's fine reach in 4h tiles, or -1 with two-level off: the
      // two-level branch in sampleVelocityComponent is then never taken.
      // w: the retired V-claims-pressure-rows lane, always zero.
      this.twoLevelEnabled ? UNIFORM_TWO_LEVEL_FINE_REACH : -1, 0,
      this.scene.fluid.surfaceTension_N_m, c.fluidWallMode === "no-slip" ? 1 : 0, activeBodyCount, c.top === "open" ? 1 : 0,
      outlet?.x ?? 0, outlet?.y ?? 0, outlet?.z ?? 0, inflow?.radius_m ?? 0,
      inflow?.velocity_m_s.x ?? 0, inflow?.velocity_m_s.y ?? 0, inflow?.velocity_m_s.z ?? 0, this.inflowBoundary?.apertureScale ?? 0,
      inflowStrength, this.referenceVolumeCells, c.fillFraction * this.executionInfo.ny, 4,
      this.sharpeningStrength, this.sharpeningDistance,
      this.volumeDustThreshold,
      c.depthBoundary === "symmetry" ? 1 : 0,
      drop?.centre_m.x ?? 0, drop?.centre_m.y ?? 0, drop?.centre_m.z ?? 0, drop?.radius_m ?? 0,
      drop?.halfHeight_m ?? 0, this.liquidOnlyVelocityAdvection ? 1 : 0,
      // w: velocityD still holds this advance's V_face authority when pressure
      // runs. MacCormack and the stage audit both overwrite it after the store.
      this.solidVoxelScratchOffsetWords, this.sharedFaceOpenValid() ? 1 : 0,
      // The shell reach in 4h tiles, whether the extension's finest passes and
      // the far-air advection arm run on those tiles, and the transport lane
      // (reach + 8, or -1 with the live set off). All inert while two-level is off.
      this.twoLevelShellReach, this.twoLevelEnabled ? 1 : 0,
      this.twoLevelEnabled ? 1 : 0,
      this.twoLevelEnabled && this.volumeDustThreshold > 0 ? UNIFORM_TRANSPORT_REACH + 8 : -1,
      // agreement: the retired compaction, phi seed and shift lanes; clamp.
      0, 0, 0, UNIFORM_PHI_AGREEMENT_CLAMP,
      // lean.x: the no-cut-cell certificate. All three sources of a cut cell
      // are host state -- the packed voxel mask, the terrain heightfield and
      // the live body list -- so this is decided here once a step rather than
      // probed per sample. See uvSolidFree.
      uniformAbOn("solidfreetrace") && this.solidVoxelsEmpty
        && !sceneHasTerrain(this.scene) && activeBodyCount === 0 ? 1 : 0,
      // lean.yzw and splash: retired census window, phi far-air arm, per-tile
      // reach and splash-survival lanes, always zero.
      0, 0, 0,
      0, 0, 0, 0,
      // splashB: cubic phi advection, retired seed cells, ghost-phi drain,
      // retired airborne momentum.
      this.geometricVolume && this.phiCubicAdvection ? 1 : 0, 0, this.geometricVolume && this.phiDrain ? 1 : 0, 0,
      this.geometricVolume ? this.orphanDustThreshold : 0, 0, 0, 0,
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
    this.mixedFrame?.invalidateExtension();
   
    if (!(ball.radius_m > 0)) return;
    this.pendingDrop = ball;
  }

  /** The editor's water shapes, as far as the one-step drop source reaches: a ball, added. */
  async editFluid(edit: LiveFluidEdit): Promise<LiveFluidEditResult> {
    this.mixedFrame?.invalidateExtension();
   
    if (edit.operation !== "add" || edit.shape !== "ball") {
      return { accepted: false, reason: "This fluid method adds water as balls only; choose Sparse Geometric to remove water or drop other shapes." };
    }
    const { width_m, height_m, depth_m } = this.scene.container, { nx, ny, nz } = this.executionInfo;
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
    // The renderer calls this every frame. Identical controls must not retire
    // geometry/support caches, the census extension, or rewrite the pressure
    // hierarchy's uniforms, nor drain the frame pipeline to apply nothing.
    const previous=this.deferredFrameValues??this.pendingPipelineValues??this.appliedRuntimeValues, keys=Object.keys(values);
    if(previous && keys.length===Object.keys(previous).length
      && keys.every(key=>Object.hasOwn(previous,key)&&Object.is(previous[key],values[key]))) {
      return;
    }
    if (this.mixedFrameInFlight) { this.deferredFrameValues = {...values}; return; }
    this.pendingPipelineValues = undefined;
    if (this.mixedFrame) {
      // Detail settings whose pipelines are not built wait (compile by need).
      const detail = uniformDetailSettings({...uniformDetailValues(this.detail), ...Object.fromEntries(UNIFORM_DETAIL_PARAM_KEYS.filter(k => values[k] !== undefined).map(k => [k, values[k]]))});
      if (!sameUniformDetailSettings(detail, this.detail) && this.awaitPipelines(this.pipelineNeedsOf({detail, relayout: true}))) { this.pendingPipelineValues = {...values}; return; }
    }
    this.mixedFrame?.invalidateExtension();
    this.faceAuthorityStored = false;
    // The renderer reapplies the latest values before the next advance.
    const finite = (key: string, fallback: number, minimum: number, maximum: number) => {
      const value = Number(values[key]);
      return Number.isFinite(value) ? Math.min(maximum, Math.max(minimum, value)) : fallback;
    };
    this.pressureMultigrid.setResidualTolerance(finite("pressureResidualTolerance", UNIFORM_PRESSURE_RESIDUAL_TOLERANCE, 0, 100));
    // Switching to "fixed" mid-run restores the full encoded schedule on the
    // next step; switching back drops the previous demand sample and uses
    // the startup budget.
    const lagged = values.pressureCycleBudget !== "fixed";
    if (lagged !== this.pressureCycleBudgetLagged) this.pressureCyclesExecutedSample = undefined;
    this.pressureCycleBudgetLagged = lagged;
    this.pressureBudgetHeadroom = Math.round(finite(
      "pressureBudgetHeadroom", this.geometricVolume ? 0 : UNIFORM_CM11A_DEFAULT_BUDGET_HEADROOM, 0, 4));
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
    this.sharpeningDistance = finite("sharpeningDistance", 2.1, this.geometricVolume ? 0 : 0.1, 3.1);
    this.solidExcessCorrection = values.solidExcessCorrection !== "off";
    this.rigidCoupling = values.rigidCoupling !== "off";
    this.fixedStep_s = uniformFixedStep_s(values.timeStep);
    if(values.velocityTransport === "maccormack" && this.velocityC.width===1 && this.executionInfo.nx>1)
      throw new Error("Changing velocity transport requires rebuilding the Uniform Geometric solver");
    this.velocityTransport = values.velocityTransport === "maccormack"
      ? "maccormack" : "semi-lagrangian";
    this.liquidOnlyVelocityAdvection = values.liquidOnlyVelocityAdvection === "on";
    // Partial-values callers must not reset the sweep budget.
    if (values.extensionFrontSweeps !== undefined && Number(values.extensionFrontSweeps) !== this.velocityExtrapolator.frontPasses) {
      this.velocityExtrapolator.setFrontPasses(Number(values.extensionFrontSweeps));
      this.publishUniformPipelineFacts();
    }
    if (this.geometricVolume) {
      // Absent means "leave as constructed".
      if (values.volumeDustThreshold !== undefined) this.volumeDustThreshold = finite("volumeDustThreshold", 0, 0, 1);
      if (values.orphanDustThreshold !== undefined) this.orphanDustThreshold = finite("orphanDustThreshold", 0, 0, 0.05);
      if (values.sharpeningSweeps !== undefined) this.sharpeningSweeps = uniformGeometricSharpeningSweeps(values.sharpeningSweeps);
      if (values.surfaceVolumeRounds !== undefined) this.surfaceVolumeRounds = uniformGeometricSurfaceVolumeRounds(values.surfaceVolumeRounds);
      // Absent detail keys keep their current settings.
      const detail = uniformDetailSettings({...uniformDetailValues(this.detail), ...Object.fromEntries(UNIFORM_DETAIL_PARAM_KEYS.filter(k => values[k] !== undefined).map(k => [k, values[k]]))});
      if (!sameUniformDetailSettings(detail, this.detail)) {
        // Solid contact changes on a running state: the remap hands a cut
        // tile to a 4h owner with its plane at the corners the h rule buried,
        // and splits one back by its cells' open fractions.
        const accepted = this.detail;
        this.detail = detail;
        this.mixedCoarsening = detail.policy === "dynamic" ? "dynamic" : "regions";
        // The relayout follows from the next frame's head (time is kept):
        // Dynamic's static key sees the change, a requests-only layout
        // takes its request here.
        // Refused: the accepted generation keeps running, under the settings
        // its frames were built for. The request is withdrawn (asked again
        // only when it is made again); its rejection stays reported.
        if (!this.syncMixedLayout()) { this.detail = accepted; this.mixedCoarsening = accepted.policy === "dynamic" ? "dynamic" : "regions"; this.mixedRefusedKey = ""; }
      } else if (this.executionInfo.uniformDetail?.rejected && UNIFORM_DETAIL_PARAM_KEYS.some(k => values[k] !== undefined)) {
        // The request is the running generation again: nothing is refused.
        this.executionInfo.uniformDetail = { ...this.executionInfo.uniformDetail, rejected: undefined };
      }
      if (values.coarseningReach !== undefined) this.mixedCoarseningReach = Math.round(finite("coarseningReach", 0, 0, 8));
      if (values.coarseningHysteresis !== undefined) this.mixedCoarseningHysteresis = Math.round(finite("coarseningHysteresis", 0, 0, 4));
      if (values.surfaceDeficitBalancing !== undefined) this.surfaceDeficitBalancing = this.geometricVolume && values.surfaceDeficitBalancing === "on";
      if (values.totalSurfaceVolume !== undefined) this.totalSurfaceVolume = values.totalSurfaceVolume === "on";
      if (values.phiCubicAdvection !== undefined) this.phiCubicAdvection = values.phiCubicAdvection === "on";
      if (values.phiDrain !== undefined) this.phiDrain = values.phiDrain === "on";
      if (values.phiPreserveSurface !== undefined) this.phiPreserveSurface = values.phiPreserveSurface === "on";
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
    this.appliedRuntimeValues={...values};
  }

  /** Pays the reduction the last step skipped, from the fields it left behind. */
  private encodeOwedDiagnosticsReduction(encoder: GPUCommandEncoder): boolean {
    if (!this.diagnosticsReductionOwed || !this.pipelines) return false;
    this.diagnosticsReductionOwed = false;
    this.run(encoder, "Uniform diagnostics reduction", this.pipelines.reduce, this.reductionGroup);
    return true;
  }

  private initializeVolumeAndTerrain(): void {
    const { nx, ny, nz } = this.executionInfo;
    const c = this.scene.container;
    const {volume,terrain,initial,wetMinimum,wetMaximum,dam} = uniformInitialVolume(this.scene,[nx,ny,nz],this.geometricVolume);
    const cellHeight = c.height_m / ny;
    if (this.vertexPhiField && this.vertexPhiScratch) {
      // Cut 4h solid owners read their plane at corners the h rule buries.
      const phi = uniformVolumeInitialPhi(this.scene, [nx, ny, nz], undefined, uniformDetailCoarseSolids(this.detail));
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
    const zeroBoundaryVelocity = new Float32Array(this.executionNegativeBoundaryVelocityBytes / 4);
    this.device.queue.writeBuffer(this.boundaryVelocityA, 0, zeroBoundaryVelocity);
    this.device.queue.writeBuffer(this.boundaryVelocityB, 0, zeroBoundaryVelocity);
    this.device.queue.writeBuffer(this.boundaryVelocityC, 0, zeroBoundaryVelocity, 0, this.boundaryVelocityC.size / 4);
    this.device.queue.writeBuffer(this.boundaryVelocityD, 0, zeroBoundaryVelocity);
    this.referenceVolumeCells = initial;
    const terrainCells = Float32Array.from(terrain, (height) => height / cellHeight);
    this.upload2DF32(this.terrainTexture, terrainCells, nx, nz);
    this.mixedTerrainCells = terrainCells;
    this.initializeActiveRegion(wetMinimum, wetMaximum);
    Object.assign(this.executionInfo, {
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
    const dimensions = [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz];
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
    // The CM11a lattice is always the domain: origin zero, window mode off.
    this.writePressureLatticeHeader([0, 0, 0], dimensions, false);
  }

  private copyField(encoder: GPUCommandEncoder, source: GPUImageCopyTexture, destination: GPUImageCopyTexture, size: GPUExtent3D): void {
    if(this.fieldPages) this.fieldPages.copy(encoder,source.texture,destination.texture);
    else encoder.copyTextureToTexture(source,destination,size);
  }

  /** Initialization hook for manufactured GPU boundary fixtures. Public textures
   * are read-only presentation adapters when fields use page storage. */
  initializeVelocityForQA(values: Float32Array): void {
    this.mixedFrame?.invalidateExtension();
    if(this.lastTime!==0 || values.length!==4*this.executionInfo.cellCount)throw new Error("Velocity fixture must initialize the complete field at t=0");
    this.upload3DF32(this.velocityA,values,this.executionInfo.nx,this.executionInfo.ny,this.executionInfo.nz);
  }

  private upload3DF32(texture: GPUTexture, values: Float32Array, nx: number, ny: number, nz: number): void {
    const field=uniformDetailField(texture);
    if(field) { field.storage.upload(texture,values); return; }
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
    if (this.windowMainGroups) {
      pass.dispatchWorkgroups(...this.windowMainGroups);
    } else if (this.activeRegionEnabled) {
      pass.dispatchWorkgroupsIndirect(this.activeDispatch, UNIFORM_ACTIVE_MAIN_DISPATCH_OFFSET);
    } else pass.dispatchWorkgroups(
      Math.ceil(this.executionInfo.nx / 4), Math.ceil(this.executionInfo.ny / 4), Math.ceil(this.executionInfo.nz / 4));
  }

  private sharedFaceOpenValid(): boolean {
    return this.geometricVolume && uniformAbOn("facecache") && !this.activeRegionEnabled
      && this.velocityTransport !== "maccormack" && !this.symmetryStageAuditFields;
  }

  /** `directory`: the detail storage's group 1 (a packed t=0 kernel). */
  private run(encoder: GPUCommandEncoder, label: string, pipeline: GPUComputePipeline, group: GPUBindGroup, directory?: GPUBindGroup): void {
    const pass = encoder.beginComputePass({ label });
    if (directory) pass.setBindGroup(1, directory);
    this.dispatch(pass, pipeline, group);
    pass.end();
  }

  private runDirect(encoder: GPUCommandEncoder, label: string, pipeline: GPUComputePipeline,
    group: GPUBindGroup, workgroups: readonly [number, number, number], directory?: GPUBindGroup): void {
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    if (directory) pass.setBindGroup(1, directory);
    pass.dispatchWorkgroups(...workgroups); pass.end();
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
    const dims = [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz];
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
    }
    const dense = lagged === undefined || this.windowForcedDenseSteps > 0;
    if (this.windowForcedDenseSteps > 0) this.windowForcedDenseSteps -= 1;
    if (dense) this.windowDenseSteps += 1;
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

  /** Level dimensions the shared active-region table describes: the pressure hierarchy's. */
  private get activeLevelDimensions(): readonly (readonly [number, number, number])[] {
    return this.pressureMultigrid.levelPhysicalDimensions;
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
          this.windowViolationAxes = words[17]!;
        }
        this.windowViolations = violations;
      } finally {
        if (buffer.mapState === "mapped") buffer.unmap();
        this.windowReadbackPending = false;
      }
    }).catch(() => { this.windowReadbackPending = false; });
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
        [Math.ceil(this.executionInfo.nx / 4), Math.ceil(this.executionInfo.ny / 4), Math.ceil(this.executionInfo.nz / 4)]);
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
      && !this.windowMainGroups && !this.activeRegionEnabled;
    if (faceAuthorityStable && this.faceAuthorityStored) this.run(encoder, "Uniform Sec. 3.3 rho-prime authority",
      this.pipelines.extrapolationDensityAuthority, this.extrapolationAuthorityGroup);
    else this.run(encoder, "Uniform Sec. 3.3 rho-prime and face authority",
      this.pipelines.extrapolationAuthority, this.extrapolationAuthorityGroup);
    this.faceAuthorityStored = faceAuthorityStable;
    if (!predicted && this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.surfaceA },
      { texture: this.symmetryStageAuditFields.extrapolationDensityAuthority },
      [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
    );
    if (!predicted) seam?.(UNIFORM_ADVANCE_PHASE.extensionAuthority);
    this.velocityExtrapolator.encode(encoder, predicted, !predicted && seam ? ((stage) => seam(
      stage === "narrow-band-front" ? UNIFORM_ADVANCE_PHASE.extensionFront
        : UNIFORM_ADVANCE_PHASE.extensionHierarchy,
    )) : undefined, false, false);
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
    this.executionInfo.uniformPipelineFacts = {
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
    Object.assign(this.executionInfo, {
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
        Object.assign(this.executionInfo, {
          uniformPressureCyclesExecuted: this.pressureCyclesExecutedSample,
          uniformPressureCyclesConverged: this.pressureCycleConvergedSample,
        });
      } finally {
        if (buffer.mapState === "mapped") buffer.unmap();
        this.pressureCycleDemandPending = false;
      }
    }).catch(() => { this.pressureCycleDemandPending = false; });
  }

  /** Mixed layout views (band reasons, sampler certificate, the frame's
   * starting tiles) are recorded only while a layer draws them. */
  private layoutViews = false;
  setLayoutViewsEnabled(enabled: boolean): void {
    if (enabled === this.layoutViews) return;
    this.layoutViews = enabled;
    this.mixedFrame?.setLayoutViews(enabled);
  }

  /**
   * The active-region header this step's finalize wrote, for the solve-window
   * view. Withdrawn while the window is off, when the dense schedule dispatches
   * the whole domain and the header is only its t=0 seed.
   */
  get solveWindowSource(): GPUFluidSolveWindowSource | undefined {
    return this.activeRegionEnabled ? { records: { buffer: this.activeRegion } } : undefined;
  }

  /** Two-level sampling is geometric and dimensionally possible. */
  private get twoLevelEnabled(): boolean {
    return this.twoLevelVelocity && this.twoLevelTileCount > 0;
  }

  private createPageAwareGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup {
    const group = this.fieldPages ? this.fieldPages.createBindGroup(descriptor) : this.device.createBindGroup(descriptor);
    this.groupDescriptors.set(group,descriptor);
    return group;
  }

  get pressureSmoothingWorkSourceForQA(): WebGPUUniformPressureMultigrid["smoothingWorkSource"] { return this.pressureMultigrid.smoothingWorkSource; }

  /** Mixed frames run ahead of their receipts: the pipeline admits another
   * frame unless UNIFORM_MIXED_RECEIPT_RING receipts are unchecked or a
   * deferred edit is draining the pipeline. Each frame adopts its own
   * layout at its head, so no census gates admission. */
  get framePending(): boolean {
    return this.mixedFramesInFlight >= UNIFORM_MIXED_RECEIPT_RING
      || (this.mixedFrameInFlight && this.deferredFrameEdit);
  }
  /** Scene, value and body edits wait for every submitted frame. */
  private get mixedFrameInFlight(): boolean { return this.mixedFramesInFlight > 0; }
  private get deferredFrameEdit(): boolean {
    return this.deferredFrameScene !== undefined || this.deferredFrameValues !== undefined || this.deferredFrameBodies !== undefined;
  }
  get deferredFramePublication(): boolean {
    return !!this.mixedFrame;
  }
  get presentationPending(): boolean { return this.framePending || this.pressureFrameFailure !== undefined; }
  async awaitFrameCompletion(): Promise<void> {
   
    await this.mixedFrameChain;
    if (this.pressureFrameFailure) throw this.pressureFrameFailure;
    await this.device.queue.onSubmittedWorkDone();
  }
  async assertSimulationHealthy(completion?: Promise<void>): Promise<void> {
    // Presentation already awaited this frame. A second queue-wide fence here
    // includes newer presentations and holds their predecessor's throughput slot.
    await (completion ?? this.awaitFrameCompletion());
  }

  advanceTo(time_s: number, bodies: RigidBodyState[] = []): boolean {
    if(this.disposed)return false;
    if (this.framePending || this.pressureFrameFailure) return false;
    if(this.mixedFrame){
      this.assertMixedOptions();
      // Reject unsupported geometry before advancing the clock or consuming a
      // queued liquid edit. The last accepted frame remains usable for edits.
      // Static voxels, terrain and non-box vessels are fine-owner solids;
      // live voxel edits relayout (editMixedSolids); rigid bodies are solid
      // geometry of the frame's solid library, promoted and coupled per step
      // (syncMixedLayout, UniformMixedBodies).
    }
    if(this.geometricVolume&&!this.mixedFrame)
      throw new Error("Uniform Geometric advances only through the mixed-ownership frame");
    // A first body waits for its pipelines: the accepted state advances without it.
    const bodiesHeld = !!this.mixedFrame && bodies.length > 0 && this.awaitPipelines(this.pipelineNeedsOf({bodies: Math.min(bodies.length, 12)}));
    // Advancing the accepted state while a change waits is the presenting
    // host's choice (acceptPipelineWaits); any other caller must not step a
    // state it did not ask for.
    if (this.mixedFrame && this.pipelineWaiting.size && !this.pipelineWaitsAccepted)
      throw new Error(`Uniform pipelines: a change waits for its ${this.pipelinePreparing} pipelines; await pipelinesPrepared() before advancing`);
    // The paper's method is calibrated for its own large-step regime (dt=1/30
    // in every Sec. 4 example): sharpening opposes per-resample transport
    // blur, so far smaller scene steps structurally out-diffuse it.
    // A paper advance is exactly 1/30 s. Treating that value only as maxDt
    // let browser callers feed 4 ms targets, producing a different 250 Hz
    // method than the Dawn/paper lane and overwhelming sharpening with many
    // extra resamples. Accumulate target time until one complete paper step
    // is available; never encode a fractional paper step.
    if (this.fixedStep_s !== undefined && !uniformFixedAdvanceReady(time_s, this.lastTime, this.fixedStep_s)) return false;
    const advance = planGPUAdvance(time_s, this.lastTime, this.fixedStep_s ?? this.scene.numerics.maxDt_s);
    if (!advance) return false;
    if (!this.pipelines) throw new Error("Uniform reference pipelines are not initialized");
    const pipelines = this.pipelines;
    const dt = advance.dt_s;
    this.lastTime = advance.nextTime_s;
    this.executionInfo.submittedTime_s = this.lastTime;
    if (!this.deferredFramePublication) this.executionInfo.simulatedTime_s = this.lastTime;
    this.executionInfo.simulationLag_s = advance.lag_s;
    this.executionInfo.lastDt_s = dt;
    this.executionInfo.lastSubsteps = 1;
    this.executionInfo.encodedSteps = (this.executionInfo.encodedSteps ?? 0) + 1;
    const roster = bodies.slice(0, 12);
    const activeBodies = bodiesHeld ? [] : this.geometricVolume ? this.prescribedSolidMotion.sample(roster, dt) : roster;
    this.stepBodyCount = activeBodies.length;
    this.rigidSystem.syncBodies(activeBodies);
    const c = this.scene.container;
    const inflow = this.scene.fluid.inflow;
    const strength = inflow ? averageInflowStrength(inflow, this.lastTime - dt, this.lastTime) : 0;
    if (this.inflowBoundary && strength > 0) {
      const cellVolume = c.width_m * c.height_m * c.depth_m
        / (this.executionInfo.nx * this.executionInfo.ny * this.executionInfo.nz);
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
        / (this.executionInfo.nx * this.executionInfo.ny * this.executionInfo.nz);
      const dropped = drop.halfHeight_m !== undefined
        ? Math.PI * drop.radius_m ** 2 * 2 * drop.halfHeight_m
        : (4 / 3) * Math.PI * drop.radius_m ** 3;
      this.referenceVolumeCells += dropped / cellVolume;
    }
    this.executionInfo.referenceLiquidVolume_cells = this.referenceVolumeCells;
    if(this.mixedFrame){
      const solids=!this.solidVoxelsEmpty||sceneHasTerrain(this.scene)||activeBodies.length>0,needs=this.pipelineNeeds;
      // A scene that lost its last solid keeps the gated variants (exact, only
      // slower) until its solid-free twins are built in the background.
      if(!solids&&!needs.holds("solidFree")&&!this.solidFreeWarming){this.solidFreeWarming=true;needs.prepare(["solidFree"],UNIFORM_PIPELINE_WARM_PACE).catch(error=>this.failMixedFrame(error));}
      this.mixedFrame.solid!.present=solids||!needs.holds("solidFree");
      const unbuilt=needs.missing(this.pipelineNeedsOf({bodies:activeBodies.length}));
      if(unbuilt.length){this.failMixedFrame(new Error(`Uniform pipelines: the frame would dispatch ${unbuilt.join(", ")} pipelines that were not prepared`));return true;}
      // The layout's owner for this roster: the census follows the bodies' GPU poses.
      try{this.syncMixedLayout(activeBodies,dt,true);this.detailStorage?.settle();}catch(error){this.failMixedFrame(error);return true;}
    }
    this.writeParams(dt, activeBodies.length, strength, drop);
    if(this.mixedFrame){
      const frame=this.mixedFrame;
      // Dynamic coarsening: the frame's head classifies the state it starts
      // from for this one step and adopts the generation it builds before
      // advecting (horizon one); moving bodies sweep this step's tiles into
      // that census. The tail extends the final velocity for the next head.
      const mixedBodies=this.mixedBodies!,bodyCount=activeBodies.length;
      this.mixedDynamic?.setBodies(bodyCount?(encoder,tiles)=>mixedBodies.encodeTiles(encoder,tiles,dt):undefined);
      // Bodies: the frame rebuilds its solid record and displaces liquid out
      // of the cells they entered; once more after the last one leaves. Native
      // coupleRigid then the rigid integration run after the projection.
      const cellVolume=c.width_m*c.height_m*c.depth_m/(this.executionInfo.nx*this.executionInfo.ny*this.executionInfo.nz);
      const frameBodies=bodyCount||this.mixedHadBodies?{couple:bodyCount&&this.rigidCoupling?(encoder:GPUCommandEncoder)=>{
        mixedBodies.encodeCoupling(encoder);this.rigidSystem.encode(encoder,dt,cellVolume,1,c.height_m/this.executionInfo.ny);
        frame.solid!.encodeBodies(encoder);
      }:undefined}:undefined;
      this.mixedHadBodies=bodyCount>0;
      let receipt:Promise<UniformMixedFrameReceipt>,kick:Promise<unknown>|undefined;
      try{
        const p={dt,gravity:this.scene.fluid.gravity_m_s2.y,density:this.scene.fluid.density_kg_m3,
        viscosity:this.scene.fluid.dynamicViscosity_Pa_s,surfaceTension:this.scene.fluid.surfaceTension_N_m,
        openTop:this.scene.container.top==="open",noSlip:this.scene.container.fluidWallMode==="no-slip",cubic:this.phiCubicAdvection,drain:this.phiDrain,preserve:this.phiPreserveSurface,
        dust:this.volumeDustThreshold,orphanDust:this.orphanDustThreshold,sharpeningStrength:this.densitySharpening?this.sharpeningStrength:0,
        sharpeningDistance:this.sharpeningDistance,sharpeningSweeps:this.sharpeningSweeps,surfaceVolumeRounds:this.surfaceVolumeRounds,pressureTolerance:this.pressureMultigrid.residualTolerance,
        totalSurfaceVolume:this.totalSurfaceVolume,redistance:this.geometricRedistance,sharpening:this.densitySharpening,
        surfaceDeficitBalancing:this.surfaceDeficitBalancing,extensionSweeps:this.velocityExtrapolator.frontPasses,
        // Impacts raise pressure demand faster than a plan two frames old
        // follows. One additional reserve covers long-dam frame 79's third
        // cycle under a 4h surface, and the dam 128^3 all-fine far-wall
        // impact's fourth (8 of 8 perturbed runs through 60 frames with it,
        // 7 of 8 without), without encoding the seven-slot envelope.
        pressureReserve:1,
        coarseSurfaceTravel:uniformDetailCoarseSurface(this.detail),coarseHeldDistance:uniformDetailHeldDistance(this.detail),
        supportPolicy:{fineReach:UNIFORM_TWO_LEVEL_FINE_REACH,shellReach:this.twoLevelShellReach,twoLevel:this.twoLevelEnabled,shellOnly:this.twoLevelEnabled},
      };
        // The seed is t=0's velocity; the drift reads it half a step on. The
        // kick is encoded ahead of the first frame with a frame's head (its
        // census, where one is due), so that frame's census bounds the
        // kicked field. Its receipt holds a ring slot until it is checked.
        if(!this.mixedKicked){this.mixedKicked=true;const k=frame.kick(p,dt/2,!!frameBodies);this.readMixedRelayout();this.mixedFramesInFlight++;kick=k.finally(()=>{this.mixedFramesInFlight--;});}
        receipt=frame.advance(p,this.mixedFrameTrace(),this.mixedGpuLayout&&(this.detail.policy==="dynamic"||this.mixedLive),frameBodies);
        if(kick)receipt=Promise.all([receipt,kick]).then(([frame])=>frame);
      }catch(error){kick?.catch(()=>undefined);this.mixedRelayoutRead=undefined;this.failMixedFrame(error);return true;}
      this.readMixedRelayout();
      this.mixedFramesInFlight++;
      const handled=receipt.then(receipt=>{
        if(this.disposed)return;
        Object.assign(this.executionInfo,{...(this.narrowBandFlip?{narrowBandFlipParticles:this.mixedFrame?.narrowBandFlip?.count,narrowBandFlipReseedClipped:this.mixedFrame?.narrowBandFlip?.reseedClipped,narrowBandFlipUnsupportedParticles:this.mixedFrame?.narrowBandFlip?.diagnostics.unsupported,narrowBandFlipMaxSurfaceDistance:this.mixedFrame?.narrowBandFlip?.diagnostics.afterMaxOutside}:{}),simulatedTime_s:advance.nextTime_s,completedTime_s:advance.nextTime_s,
          uniformPressureAcceptedResidual:receipt.residual,uniformPressureCyclesExecuted:receipt.cycles,uniformPressureCyclesConverged:true,
          uniformPressureCyclesEncoded:receipt.encoded,uniformPressureCyclesConfigured:this.pressureSchedule.fullCycles+this.pressureSchedule.vCycles,
          uniformVolumeDustCells:receipt.dustOwners,uniformVolumeDustMass_cells:receipt.dustMass_cells,
          uniformVolumeOrphanDustCells:receipt.orphanDustOwners,uniformVolumeOrphanDustMass_cells:receipt.orphanDustMass_cells,
          uniformPressureBandTiles:receipt.bandTiles,uniformPressureBandCycles:receipt.bandCycles,uniformPressureBandResidual:receipt.bandResidual});
        // The detail planner's clock: hysteresis and expiry count accepted frames.
        this.detailAcceptedStep++;
      }).catch(error=>this.failMixedFrame(error)).finally(()=>{
        this.mixedFramesInFlight--;
        if(this.disposed||this.pressureFrameFailure||this.mixedFrameInFlight)return;
        const scene=this.deferredFrameScene,values=this.deferredFrameValues,bodies=this.deferredFrameBodies;
        this.deferredFrameScene=undefined;this.deferredFrameValues=undefined;this.deferredFrameBodies=undefined;
        if(scene)this.applySceneUniforms(scene);if(values)this.applyRuntimeValues(values);if(bodies)this.syncRigidBodies(bodies);
      });
      this.mixedFrameChain=Promise.all([this.mixedFrameChain,handled]).then(()=>undefined);
      return true;
    }

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
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.velocityA }, { texture: this.symmetryStageAuditFields.preExtrapolationVelocity },
      [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
    );
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.volumeA }, { texture: this.symmetryStageAuditFields.previousRawDensity },
      [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
    );
    if (this.symmetryStageAuditNegativeBoundaryVelocity) encoder.copyBufferToBuffer(
      this.boundaryVelocityA, 0, this.symmetryStageAuditNegativeBoundaryVelocity, 0,
      this.executionNegativeBoundaryVelocityBytes,
    );
    this.encodeVelocityExtrapolation(encoder, false, seam);

    // Sec. 3.6 must see the updated solid geometry before Sec. 3.4 excludes
    // solid donors. Without this pair, density in cells newly covered by a
    // moving body is skipped by beta construction and then overwritten with
    // zero by the gather, so the supposedly conservative operator loses the
    // displaced liquid before the historical post-sharpening cleanup runs.
    if (this.solidExcessCorrection && (activeBodies.length > 0 || sceneHasTerrain(this.scene))) {
      // Ranged to the donor-sum region for the same reason the transport
      // clears are: the scatter/resolve pair only addresses [0,N), and words
      // [N,2N) carry the 4h classes for the rest of the step. Geometric mode
      // has already reconciled covered water before classifying those tiles.
      encoder.clearBuffer(this.conditioningScratch, 0, this.executionInfo.nx * this.executionInfo.ny * this.executionInfo.nz * 4);
      this.run(encoder, "Uniform moving-solid entry excess scatter",
        pipelines.scatterSolidExcess, this.solidEntryScatterGroup);
      this.run(encoder, "Uniform moving-solid entry excess resolve",
        pipelines.resolveSolidExcess, this.solidEntryResolveGroup);
    }

    // Algorithm 1 steps 1-2, paper Secs. 3.3-3.5: use the extrapolated
    // current velocity for the modified conservative semi-Lagrangian density
    // operator, diffuse gamma in each dimension, then sharpen locally.
    encoder.clearBuffer(this.conditioningScratch);
    this.run(encoder, "Uniform trace gamma and beta", pipelines.traceGammaBeta, this.densityTraceGroup);
    if (this.symmetryStageAuditBetaBuffer) encoder.copyBufferToBuffer(
      this.conditioningScratch, 0, this.symmetryStageAuditBetaBuffer, 0,
      this.executionInfo.nx * this.executionInfo.ny * this.executionInfo.nz * 4,
    );
    this.run(encoder, "Uniform scatter density deficits", pipelines.scatterDensityDeficit, this.densityScatterGroup);
    this.run(encoder, "Uniform gather conservative density", pipelines.gatherDensity, this.densityGatherGroup);
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.volumeB }, { texture: this.symmetryStageAuditFields.densityAdvection },
      [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
    );
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.gammaA }, { texture: this.symmetryStageAuditFields.gammaPostAdvection },
      [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
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
      ["x", pipelines.diffuseGammaX],
      ["y", pipelines.diffuseGammaY],
      ["z", pipelines.diffuseGammaZ],
    ] as const;
    for (let iteration = 0; iteration < this.gammaDiffusionIterations; iteration += 1) {
      diffusionPasses.forEach(([axis, pipeline], index) => this.run(encoder,
        `Uniform gamma diffusion iteration ${iteration + 1}/${this.gammaDiffusionIterations} ${axis} Jacobi`,
        pipeline, this.gammaDiffusionGroups[index & 1]!));
      // Three axis passes leave their result in the A/B output pair. Restore
      // the density-advection ABI expected by sharpening and by the next
      // repetition.
      this.copyField(encoder,{ texture: this.volumeA }, { texture: this.volumeB },
        [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz]);
      this.copyField(encoder,{ texture: this.gammaB }, { texture: this.gammaA },
        [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz]);
    }
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.volumeB }, { texture: this.symmetryStageAuditFields.densityDiffusion },
      [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
    );
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.gammaA }, { texture: this.symmetryStageAuditFields.gammaPostDiffusion },
      [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
    );
    if (this.gammaDiffusionIterations > 0) seam?.(UNIFORM_ADVANCE_PHASE.gammaDiffusion);
    if (this.densitySharpening) {
      encoder.clearBuffer(this.conditioningScratch);
      this.run(encoder, "Uniform interface sharpening", pipelines.sharpenCompute, this.sharpenComputeGroup);
      if (this.sharpeningMassCorrection) {
        seam?.(UNIFORM_ADVANCE_PHASE.interfaceSharpening);
        this.run(encoder, "Uniform conserved sharpening scatter", pipelines.sharpenScatter, this.sharpenScatterGroup);
        this.run(encoder, "Uniform conserved sharpening resolve", pipelines.sharpenResolve, this.sharpenResolveGroup);
        seam?.(UNIFORM_ADVANCE_PHASE.sharpeningMassCorrection);
      } else {
        // sharpenCompute writes volumeA while every downstream surface stage
        // reads volumeB. Preserve that ABI even for the deliberately
        // non-conservative one-pass ablation.
        this.copyField(encoder,
          { texture: this.volumeA }, { texture: this.volumeB },
          [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
        );
        seam?.(UNIFORM_ADVANCE_PHASE.interfaceSharpening);
      }
    }
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.volumeB }, { texture: this.symmetryStageAuditFields.densitySharpening },
      [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
    );
    if (this.solidExcessCorrection && (activeBodies.length > 0 || sceneHasTerrain(this.scene))) {
      encoder.clearBuffer(this.conditioningScratch);
      this.run(encoder, "Uniform partial-solid excess scatter", pipelines.scatterSolidExcess, this.solidExcessScatterGroup);
      this.run(encoder, "Uniform partial-solid excess resolve", pipelines.resolveSolidExcess, this.solidExcessResolveGroup);
      seam?.(UNIFORM_ADVANCE_PHASE.solidExcess);
    }
    this.copyField(encoder,{ texture: this.volumeB }, { texture: this.volumeA }, [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz]);
    // Algorithm 1 steps 3-4: advect/force velocity after the surface-density
    // update, then enforce incompressibility.
    if (this.velocityTransport === "maccormack") {
      // CM11b Sec. 3.5 modified MacCormack with local-extrema fallback. Extend
      // the forward prediction before the reverse trace so every lookup has a
      // defined velocity; body forces are applied exactly once in correction.
      this.run(encoder, "Uniform bounded MacCormack velocity prediction",
        pipelines.advect, this.advectGroup);
      if (this.symmetryStageAuditFields) this.copyField(encoder,
        { texture: this.velocityC }, { texture: this.symmetryStageAuditFields.velocityPrediction },
        [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
      );
      this.encodeVelocityExtrapolation(encoder, true);
      this.run(encoder, "Uniform bounded MacCormack reverse advection",
        pipelines.reverse, this.reverseGroup);
      // velocityD is also the opt-in reverse-advection audit texture.
      this.run(encoder, "Uniform bounded MacCormack correction and body forces",
        pipelines.correct, this.correctGroup);
    } else {
      // The original one-pass path already performs the midpoint backward
      // trace and applies body forces exactly once to the advected field.
      this.run(encoder, "Uniform semi-Lagrangian velocity advection and body forces",
        pipelines.semiLagrangian, this.semiLagrangianGroup);
      if (this.symmetryStageAuditFields) this.copyField(encoder,
        { texture: this.velocityB }, { texture: this.symmetryStageAuditFields.velocityPrediction },
        [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
      );
      // The audit schema predates the selectable transport. Publish identity
      // placeholders for its MacCormack-only intermediate stages so a
      // semi-Lagrangian run never exposes stale texture contents as evidence.
      if (this.symmetryStageAuditFields) {
        this.copyField(encoder,
          { texture: this.velocityB }, { texture: this.velocityD },
          [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
        );
        this.copyField(encoder,
          { texture: this.transportA }, { texture: this.transportB },
          [this.executionInfo.nx + 2, this.executionInfo.ny + 2, this.executionInfo.nz + 2],
        );
      }
    }
    if (this.symmetryStageAuditFields) this.copyField(encoder,
      { texture: this.velocityB }, { texture: this.symmetryStageAuditFields.velocityAdvection },
      [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
    );
    seam?.(UNIFORM_ADVANCE_PHASE.advectionCorrection);
    const pressureBoundary = seam && ((stage: UniformCM11aPlanStage) => seam(UNIFORM_PRESSURE_STAGE_PHASE[stage]));
    const finishFrame = () => {
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
      this.run(encoder, "Uniform pressure projection", pipelines.project, this.projectGroup);
      if (this.symmetryStageAuditFields) this.copyField(encoder,
        { texture: this.velocityA }, { texture: this.symmetryStageAuditFields.pressureProjection },
        [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz],
      );
      seam?.(UNIFORM_ADVANCE_PHASE.pressureProjection);
      if (this.rigidCoupling && activeBodies.length > 0) {
        this.run(encoder, "Uniform rigid-body coupling", pipelines.coupleRigid, this.rigidGroup);
        this.copyField(encoder,{ texture: this.volumeB }, { texture: this.volumeA }, [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz]);
        this.copyField(encoder,{ texture: this.velocityB }, { texture: this.velocityA }, [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz]);
        encoder.copyBufferToBuffer(this.boundaryVelocityB, 0, this.boundaryVelocityA, 0, this.executionNegativeBoundaryVelocityBytes);
        const cellVolume = c.width_m * c.height_m * c.depth_m / (this.executionInfo.nx * this.executionInfo.ny * this.executionInfo.nz);
        this.rigidSystem.encode(encoder, dt, cellVolume, 1, c.height_m / this.executionInfo.ny);
        seam?.(UNIFORM_ADVANCE_PHASE.rigidCoupling);
      }
      // Sec. 3.8 remains the optional global reconstruction. Container geometry
      // is not selected here; it is already present in SolidWorld.
      if (this.densityPostProcessing) {
        this.run(encoder, "Uniform post-process blur x", pipelines.postprocessBlurX, this.postprocessBlurXGroup);
        this.run(encoder, "Uniform post-process blur y", pipelines.postprocessBlurY, this.postprocessBlurYGroup);
        this.run(encoder, "Uniform post-process blur z", pipelines.postprocessBlurZ, this.postprocessBlurZGroup);
        this.run(encoder, "Uniform sub-grid surface resolve", pipelines.postprocessResolve, this.postprocessResolveGroup);
      } else {
        this.run(encoder, "Uniform wall-film resolve", pipelines.wallFilmResolve, this.wallFilmResolveGroup);
      }
      seam?.(UNIFORM_ADVANCE_PHASE.densityPostProcess);
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
      if (!this.diagnosticsReductionOwed) this.run(encoder, "Uniform diagnostics reduction", pipelines.reduce, this.reductionGroup);
      physicsCPUTrace?.completePhase(UNIFORM_ADVANCE_PHASE.diagnosticsReduction);
      physicsTrace?.resolve(encoder);
      physicsQueueTrace?.begin();
      this.device.queue.submit([encoder.finish()]);
      if (pressureCycleDemandEncoded) this.readPressureCycleDemand();
      if (windowReadbackEncoded) this.readWindowBox();
      if (physicsCPUTrace) {
        this.executionInfo.physicsCPUTrace = physicsCPUTrace.finish({ id: "other", label: "Capture closure + command submission" });
        this.executionInfo.physicsCaptureIdentity = {
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
            this.executionInfo.physicsTrace = trace;
          }
        }).catch(() => {}).finally(() => { this.physicsTracePending = false; });
      }
      this.executionInfo.submittedTime_s = this.lastTime;
      this.executionInfo.simulatedTime_s = this.lastTime;
      const submittedTime = this.lastTime;
      void this.device.queue.onSubmittedWorkDone().then(() => {
        if (!this.disposed) this.executionInfo.completedTime_s = Math.max(this.executionInfo.completedTime_s ?? 0, submittedTime);
      }).catch(() => {});
    };
    this.pressureMultigrid.encode(encoder, this.pressureMultigridGroup,
      pressureBoundary, this.planPressureCycleBudget(), this.pressureRecoveryExpected);
    finishFrame();
    return true;
  }

  async readStats(): Promise<GPUEulerianInfo> {
   
    await this.awaitFrameCompletion();
    this.refreshMixedAllocation();
    if(this.mixedFrame){
      if(this.disposed||this.readbackPending)return this.executionInfo;
      this.readbackPending=true;
      this.statsReadback??=this.device.createBuffer({label:"Uniform diagnostics readback",size:264,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try{
        const encoder=this.device.createCommandEncoder();this.mixedDiagnostics!.encode(encoder);
        encoder.copyBufferToBuffer(this.reductions,0,this.statsReadback,0,24);
        encoder.copyBufferToBuffer(this.mixedFrame.ownership.support,this.mixedFrame.ownership.capacity.tiles*16,this.statsReadback,24,16);
        this.device.queue.submit([encoder.finish()]);
        await this.statsReadback.mapAsync(GPUMapMode.READ);const words=new Uint32Array(this.statsReadback.getMappedRange(),0,10).slice();
        const represented=words[0]!/2048,volume=words[3]!/2048,reference=Math.max(1,this.referenceVolumeCells);
        Object.assign(this.executionInfo,{representedVolumeCellSum:represented,volumeCellSum:volume,
          representedVolumeDrift:(represented-reference)/reference,rawVolumeDrift:(volume-reference)/reference,volumeDrift:(volume-reference)/reference,
          volumeTelemetrySource:"dense-volume",frontTelemetrySource:"dense-volume",
          front_m:-this.scene.container.width_m/2+words[1]!*this.scene.container.width_m/this.executionInfo.nx,
          maxSpeed_m_s:new Float32Array(words.buffer)[2],uniformTwoLevelVelocity:this.twoLevelEnabled,
          uniformTwoLevelFineReach:UNIFORM_TWO_LEVEL_FINE_REACH,uniformTwoLevelShellReach:this.twoLevelShellReach,
          uniformTwoLevelFineTiles:words[4],uniformTwoLevelShellTiles:words[5],uniformTwoLevelTilesTotal:this.mixedFrame.ownership.capacity.tiles,
          uniformTwoLevelExtensionTiles:this.twoLevelEnabled,uniformMixedRegularTiles:words[7],uniformMixedGeneralTiles:words[8]});
        return this.executionInfo;
      }finally{if(this.statsReadback.mapState==="mapped")this.statsReadback.unmap();this.readbackPending=false;}
    }
    if (this.disposed || this.readbackPending) return this.executionInfo;
    this.readbackPending = true;
    this.statsReadback ??= this.device.createBuffer({ label: "Uniform reference diagnostics readback", size: 264, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Uniform reference diagnostics readback" });
    this.encodeOwedDiagnosticsReduction(encoder);
    encoder.copyBufferToBuffer(this.reductions, 0, this.statsReadback, 0, 32);
    encoder.copyBufferToBuffer(this.reductions, 32, this.statsReadback, 248, 8);
    encoder.copyBufferToBuffer(this.reductions, 40, this.statsReadback, 256, 8);
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
      if ((this.executionInfo.encodedSteps ?? 0) > 0) {
        this.pressureCyclesExecutedSample = words[45]! + words[46]!;
        this.pressureCycleConvergedSample = words[44] === 1;
      }
      const reference = Math.max(1, this.referenceVolumeCells);
      this.executionInfo.representedVolumeCellSum = words[0] / 2048;
      this.executionInfo.volumeCellSum = words[3] / 2048;
      this.executionInfo.representedVolumeDrift = (this.executionInfo.representedVolumeCellSum - reference) / reference;
      this.executionInfo.rawVolumeDrift = (this.executionInfo.volumeCellSum - reference) / reference;
      this.executionInfo.volumeDrift = this.executionInfo.rawVolumeDrift;
      this.executionInfo.volumeTelemetrySource = "dense-volume";
      this.executionInfo.front_m = -this.scene.container.width_m / 2
        + words[1] * this.scene.container.width_m / this.executionInfo.nx;
      this.executionInfo.frontTelemetrySource = "dense-volume";
      this.executionInfo.maxSpeed_m_s = new Float32Array(new Uint32Array([words[2]]).buffer)[0];
      Object.assign(this.executionInfo, { uniformUnplaceableSolidExcess_cells: words[4] / 2048 });
      Object.assign(this.executionInfo, {
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
      Object.assign(this.executionInfo, {
        uniformActiveRegionMinimum: activeMinimum,
        uniformActiveRegionMaximum: activeMaximum,
        uniformActiveRegionCellCount: activeCellCount,
        uniformActiveRegionFraction: activeCellCount / Math.max(1, this.executionInfo.cellCount),
        uniformSolveWindowDispatch: this.activeRegionEnabled
          ? (this.windowDispatchIndirect ? "indirect" : "host") : undefined,
        uniformSolveWindowClippedSteps: this.activeRegionEnabled && !this.windowDispatchIndirect
          ? this.windowViolations : undefined,
        uniformSolveWindowDenseSteps: this.activeRegionEnabled && !this.windowDispatchIndirect
          ? this.windowDenseSteps : undefined,
        uniformSolveWindowMaxLagSteps: this.activeRegionEnabled && !this.windowDispatchIndirect
          ? this.windowMaxLagSteps : undefined,
      });
      return this.executionInfo;
    } finally {
      if (this.statsReadback.mapState === "mapped") this.statsReadback.unmap();
      this.readbackPending = false;
    }
  }

  enableCM11aCoarsestCapture(invocation = 1): void {
    this.pressureMultigrid.enableCoarsestCapture(invocation);
  }
  readCM11aCoarsestCapture(): Promise<UniformCM11aCoarsestCapture | undefined> {
    return this.pressureMultigrid.readCoarsestCapture();
  }

  /** A voxel stroke reaches this solver as solid-mask bits, so only a different lattice is refused. */
  validateLiveSolidEdit(scene: SceneDescription): void {
    const [nx, ny] = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
    if (nx !== this.executionInfo.nx || ny !== this.executionInfo.ny) throw new Error("This voxel edit changes the fluid lattice; reset the scene to apply it.");
  }

  /** Fence paused remapping so the renderer invalidates its retained surface
   * and publishes resolution status without advancing simulation time. */
  async refreshSceneTopology():Promise<void>{
    this.mixedFrame?.invalidateExtension();
    if(!this.mixedFrame)return;
    await this.awaitFrameCompletion();await this.device.queue.onSubmittedWorkDone();
  }

  applySceneUniforms(scene: SceneDescription): void {
    this.mixedFrame?.invalidateExtension();
    this.appliedRuntimeValues = undefined;
    if (this.mixedFrameInFlight) { this.deferredFrameScene = scene; return; }
    this.pendingPipelineScene = undefined;
    // A scene whose pipelines are not built waits (compile by need): the first
    // voxel of a solid-free scene, a first live edit, a region drawn under
    // the domain placement. The accepted scene keeps advancing.
    if (this.mixedFrame && this.pipelineNeeds.deferred && this.awaitPipelines(this.pipelineSceneNeeds(scene))) { this.pendingPipelineScene = scene; return; }
    this.scene = scene;
    this.faceAuthorityStored = false;
    const dirty = this.solidMask.update(solidWorldForScene(scene));
    if (dirty) { this.solidVoxelsEmpty = uniformSolidMaskEmpty(this.solidMask); this.mixedSolidMaskStamp++; }
    // Dynamic coarsening joins the edit's tiles on the GPU (the next head
    // adopts them); a requests-only layout takes the scene's request.
    if (dirty && this.mixedFrame) this.editMixedSolids(dirty);
    this.syncMixedLayout();
    if (dirty) this.device.queue.writeBuffer(this.activeScratch,
      (this.solidVoxelScratchOffsetWords + dirty.firstWord) * 4, this.solidMask.words.buffer as ArrayBuffer,
      dirty.firstWord * 4, dirty.wordCount * 4);
    this.inflowBoundary = scene.fluid.inflow
      ? createInflowGridBoundary(scene.fluid.inflow, scene.container, [this.executionInfo.nx, this.executionInfo.ny, this.executionInfo.nz])
      : undefined;
    this.pressureMultigrid.setDepthSymmetry(scene.container.depthBoundary === "symmetry");
    this.activeRegionRescanPending = true;
  }

  syncRigidBodies(bodies: readonly RigidBodyState[]): void {
    // A first body's pipelines start here; advanceTo holds the roster until they exist.
    if (this.mixedFrame && bodies.length) this.awaitPipelines(this.pipelineNeedsOf({bodies: Math.min(bodies.length, 12)}));
    this.mixedFrame?.invalidateExtension();
    if (this.mixedFrameInFlight) { this.deferredFrameBodies = structuredClone([...bodies]); return; }
    this.rigidSystem.syncBodies(bodies);
  }
  get rigidRenderBuffer(): GPUBuffer { return this.rigidSystem.renderBuffer; }
  get rigidMotionBuffer(): GPUBuffer { return this.rigidSystem.motionBuffer; }
  setSelectedRigidBody(index: number): void { this.rigidSystem.setSelectedIndex(index); }
  pickRigidBody(origin: RigidBodyState["position_m"], direction: RigidBodyState["position_m"]) { return this.rigidSystem.pick(origin, direction); }
  readRigidBodyPoses() { return this.rigidSystem.readPoses(); }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.mixedFrame?.destroy();this.mixedPressureGeometry?.target.destroy();this.mixedPressureGeometry?.centerPhi.destroy();
    this.detailStorage?.destroy();
    this.mixedDynamic?.destroy();this.mixedBuilder?.destroy();for(const read of this.mixedRelayoutReads)read.destroy();this.mixedRelayoutRead?.destroy();this.mixedBodies?.destroy();
    for (const texture of new Set([
      this.velocityA, this.velocityB, this.velocityC, this.velocityD,
      this.pressureA, this.pressureB, this.volumeA, this.volumeB,
      this.surfaceA, this.surfaceB, this.gammaA, this.gammaB,
      ...Object.values(this.symmetryStageAuditFields ?? {}),
      this.heightA, this.heightB, this.terrainTexture,
      this.transportA, this.transportB,
    ])) texture.destroy();
    this.vertexPhiField?.destroy(); this.vertexPhiScratch?.destroy();
    if(this.scratchArena)this.scratchArena.destroy();
    else this.volumeEdges?.destroy();
    this.boundaryVelocityA.destroy(); this.boundaryVelocityB.destroy();
    this.boundaryVelocityC.destroy(); this.boundaryVelocityD.destroy();
    this.symmetryStageAuditNegativeBoundaryVelocity?.destroy();
    this.symmetryStageAuditBetaBuffer?.destroy();
    this.macCormackAuditBinding.destroy();
    this.velocityExtrapolator.destroy();
    this.pressureMultigrid.destroy();
    this.params.destroy();
    this.reductions.destroy();
    this.conditioningScratch.destroy();
    this.activeRegion.destroy();
    this.activeScratch.destroy();
    this.activeDispatch.destroy();
    this.fieldPages?.destroy();
    this.rigidSystem.destroy(); this.rigidExchange.destroy();
    this.statsReadback?.destroy();
    // An outstanding map rejects on destroy; `readPressureCycleDemand` catches
    // it and the `disposed` guard keeps it from touching a dead solver.
    this.pressureCycleDemandReadback?.destroy();
  }
}
