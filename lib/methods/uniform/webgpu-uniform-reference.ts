import {
  damBreakBoxContains,
  initialLiquidFractionAtCell,
  sceneDamBreakBox,
} from "../../core/initial-fluid";
import { averageInflowStrength, createInflowGridBoundary, type InflowGridBoundary } from "../../core/inflow-boundary";
import type { SceneDescription } from "../../core/model";
import { planUniformHostAllocation } from "./uniform-host-allocation";
import { initializeRigidBodies, type RigidBodyState } from "../../core/rigid-body";
import { sceneLatticeDimensions } from "../../core/scene-lattice";
import { planGPUAdvance } from "../../core/tall-cell-diagnostics";
import type { GPUQuality } from "../../core/gpu-quality";
import { sceneHasTerrain, terrainColumnHeights } from "../../core/terrain";
import {
  sceneHasSphericalContainer,
  sphericalContainerOpenFractionAtCell,
} from "../../core/spherical-container";
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
  UNIFORM_CM11A_FULL_CYCLES,
  UNIFORM_CM11A_POST_SWEEPS,
  UNIFORM_CM11A_PRE_SWEEPS,
  UNIFORM_CM11A_V_CYCLES,
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
import type {
  FluidPipelineContext,
  FluidPipelineGraph,
  FluidPipelineStage,
} from "../../core/fluid-pipeline";
import { UNIFORM_PAPER_DT_S, uniformPaperAdvanceReady } from "./uniform-paper";
import { packTankWallField } from "../../core/tank-wall-field";

export { UNIFORM_PAPER_DT_S } from "./uniform-paper";

export interface WebGPUUniformReferenceOptions {
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
  /** Sec. 3.6 cut-cell excess redistribution. */
  solidExcessCorrection?: boolean;
  /** Two-way fluid/body exchange and rigid integration. */
  rigidCoupling?: boolean;
  /** CM11a cycle and smoothing schedule. */
  pressureSchedule?: UniformCM11aSchedule;
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
export const UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS = 1;
export const UNIFORM_GAMMA_DIFFUSION_MAX_ITERATIONS = 7;

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
const UNIFORM_ACTIVE_SUMMARY_BYTES = 32;

/**
 * The exact partition of one uniform advance, in encode order.
 *
 * These are the trace seams `advanceTo` emits and the labels the fluid
 * pipeline panel's stage graph owns — one table serving both, declared beside
 * the encoder it describes, so a stage cannot drift from its measurement. A
 * seam closes everything since the previous seam, clears and copies included:
 * every phase is charged for the buffer state its passes depend on.
 */
export const UNIFORM_ADVANCE_PHASE = Object.freeze({
  extensionAuthority: { id: "velocity-extrapolation", label: "Sec. 3.3 interface authority" },
  extensionFront: { id: "velocity-extrapolation", label: "Sec. 3.3 narrow-band FIM front" },
  extensionHierarchy: { id: "velocity-extrapolation", label: "Sec. 3.3 hierarchy fill + transport shell" },
  densityAdvection: { id: "fine-sdf-advection", label: "Sec. 3.4 conservative density advection" },
  gammaDiffusion: { id: "fine-sdf-advection", label: "Sec. 3.4 axis-Jacobi gamma diffusion" },
  interfaceSharpening: { id: "fine-sdf-redistance", label: "Sec. 3.5 interface density correction" },
  sharpeningMassCorrection: { id: "fine-sdf-redistance", label: "Sec. 3.5 local mass return" },
  solidExcess: { id: "fine-sdf-redistance", label: "Sec. 3.6 partial-solid excess" },
  advectionCorrection: { id: "velocity-advection", label: "Velocity advection + body forces" },
  pressureSetup: { id: "pressure-system", label: "CM11a topology + RHS pyramid" },
  pressureFullCycles: { id: "pressure-solve", label: "CM11a Full-Cycles" },
  pressureVCycles: { id: "pressure-solve", label: "CM11a V-Cycles" },
  pressureFinish: { id: "pressure-solve", label: "CM11a parity copy + fine residual" },
  pressureProjection: { id: "velocity-projection", label: "Pressure projection" },
  rigidCoupling: { id: "other", label: "Rigid two-way coupling + integration" },
  densityPostProcess: { id: "surface-extraction", label: "Render-only wall-film / Sec. 3.8 reconstruction" },
  diagnosticsReduction: { id: "other", label: "Diagnostics reduction" },
} satisfies Record<string, GPUTimestampPhase>);

/** The multigrid's self-reported schedule groups, mapped onto the phase table. */
const UNIFORM_PRESSURE_STAGE_PHASE: Readonly<Record<UniformCM11aPlanStage, GPUTimestampPhase>> = Object.freeze({
  setup: UNIFORM_ADVANCE_PHASE.pressureSetup,
  "full-cycle": UNIFORM_ADVANCE_PHASE.pressureFullCycles,
  "v-cycle": UNIFORM_ADVANCE_PHASE.pressureVCycles,
  finish: UNIFORM_ADVANCE_PHASE.pressureFinish,
});

/**
 * Standalone dense reference implementation.
 *
 * This class deliberately has no octree option or adaptive compatibility
 * branch. Its allocations, pipelines, step graph, and diagnostics are owned
 * entirely by the `uniform` method plugin.
 */
export class WebGPUUniformReferenceSolver implements GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  get surfaceFieldTexture(): GPUTexture {
    return this.surfaceB;
  }
  readonly columnBaseTexture: GPUTexture;
  readonly velocityTexture: GPUTexture;
  /** Velocity after advection/forces and before the pressure projection. */
  readonly preProjectionVelocityTexture: GPUTexture;
  /** Padded float32 velocity extension retained for opt-in comparison diagnostics. */
  readonly extrapolatedVelocityTexture: GPUTexture;
  /** FIM xyz distances plus the 3-bit active mask in w, for Dawn conformance diagnostics. */
  readonly extrapolationActiveStateTexture: GPUTexture;
  readonly extrapolationActiveFrontPassCeiling: number;
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
  private readonly params: GPUBuffer;
  private readonly tankWallScratchOffsetWords: number;
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
  private solidExcessCorrection: boolean;
  private rigidCoupling: boolean;
  private readonly pressureSchedule: UniformCM11aSchedule;
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

  private constructor(
    private readonly device: GPUDevice,
    public scene: SceneDescription,
    quality: GPUQuality,
    _onRigidLoads?: (loads: GPURigidLoad[]) => void,
    options: WebGPUUniformReferenceOptions = {},
  ) {
    this.activeRegionEnabled = options.activeRegion === true
      && (typeof process === "undefined" || process.env.FLUID_UNIFORM_ACTIVE_REGION !== "0");
    this.densityPostProcessing = options.densityPostProcessing === true;
    this.densitySharpening = options.densitySharpening !== false;
    this.sharpeningMassCorrection = options.sharpeningMassCorrection !== false;
    this.gammaDiffusionIterations = Math.round(Math.min(UNIFORM_GAMMA_DIFFUSION_MAX_ITERATIONS,
      Math.max(0, options.gammaDiffusionIterations ?? UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS)));
    this.sharpeningStrength = Math.min(2, Math.max(0.25, options.sharpeningStrength ?? 1));
    this.sharpeningDistance = Math.min(3.1, Math.max(0.1, options.sharpeningDistance ?? 2.1));
    this.solidExcessCorrection = options.solidExcessCorrection !== false;
    this.rigidCoupling = options.rigidCoupling !== false;
    this.pressureSchedule = options.pressureSchedule ?? {
      fullCycles: UNIFORM_CM11A_FULL_CYCLES,
      vCycles: UNIFORM_CM11A_V_CYCLES,
      preSweeps: UNIFORM_CM11A_PRE_SWEEPS,
      postSweeps: UNIFORM_CM11A_POST_SWEEPS,
    };
    this.paperTimeStep = options.timeStep !== "scene";
    this.velocityTransport = options.velocityTransport === "maccormack"
      ? "maccormack" : "semi-lagrangian";
    this.liquidOnlyVelocityAdvection = options.liquidOnlyVelocityAdvection === true;
    // The stage trace's closing marker pass must dispatch observable work, or
    // Metal skips its end-of-pass timestamp and the first sample retires
    // hardware tracing for this solver. Compile it long before the panel asks.
    void GPUStageTimestampRecorder.prepare(device);
    const [nx, ny, nz] = sceneLatticeDimensions(scene, device.limits.maxTextureDimension3D);
    const allocation = planUniformHostAllocation(nx, ny, nz, "maccormack");
    this.negativeBoundaryVelocityBytes = allocation.boundaryVelocityBytes;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const texture3d = (label: string, format: GPUTextureFormat, size: GPUExtent3D) =>
      device.createTexture({ label, size, dimension: "3d", format, usage });
    const velocity = (label: string) => texture3d(label, "rgba32float", allocation.velocityExtent);
    const scalar = (label: string) => texture3d(label, "r32float", allocation.volumeExtent);
    this.velocityA = velocity("Uniform reference velocity A");
    this.velocityB = velocity("Uniform reference velocity B");
    this.velocityC = velocity("Uniform reference velocity C");
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
    this.pressureA = scalar("Uniform reference pressure A");
    this.pressureB = scalar("Uniform reference pressure B");
    this.volumeA = scalar("Uniform reference volume A");
    this.volumeB = scalar("Uniform reference volume B");
    this.surfaceA = scalar("Uniform reference smoothed surface A");
    this.surfaceB = scalar("Uniform reference smoothed surface B");
    this.gammaA = scalar("Uniform reference transport gamma A");
    this.gammaB = scalar("Uniform reference transport gamma B");
    this.heightA = device.createTexture({ label: "Uniform reference column base", size: [nx, nz], format: "rg32float", usage });
    this.heightB = device.createTexture({ label: "Uniform reference column occupancy", size: [nx, nz], format: "rg32float", usage });
    this.terrainTexture = device.createTexture({ label: "Uniform reference terrain", size: [nx, nz], format: "r32float", usage });
    const transportUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    this.transportA = device.createTexture({ label: "Uniform reference transport A", size: allocation.transportExtent, dimension: "3d", format: "rgba32float", usage: transportUsage });
    this.transportB = device.createTexture({ label: "Uniform reference transport B", size: allocation.transportExtent, dimension: "3d", format: "rgba32float", usage: transportUsage });
    if (typeof process !== "undefined" && process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT === "1") {
      this.symmetryStageAuditTextures = Object.freeze({
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
    this.params = device.createBuffer({ label: "Uniform reference parameters", size: 176, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const packedTankWalls = packTankWallField(scene.container.wallField);
    const activeRegionBytes = (UNIFORM_ACTIVE_LEVEL_BASE_WORD
      + UNIFORM_ACTIVE_MAX_LEVELS * UNIFORM_ACTIVE_LEVEL_WORDS) * 4;
    this.activeRegion = device.createBuffer({
      label: "Uniform reference active liquid region", size: activeRegionBytes,
      // readStats copies the published bounds/counters into its readback
      // packet in both dense and sparse modes.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const activeSummaryCount = Math.ceil(nx / 4) * Math.ceil(ny / 4) * Math.ceil(nz / 4);
    const activeSummaryBytes = activeSummaryCount * UNIFORM_ACTIVE_SUMMARY_BYTES;
    this.tankWallScratchOffsetWords = (activeRegionBytes + activeSummaryBytes) / 4;
    this.activeScratch = device.createBuffer({
      label: "Uniform reference active liquid census scratch and summaries",
      size: activeRegionBytes + activeSummaryBytes + packedTankWalls.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.activeScratch, this.tankWallScratchOffsetWords * 4, packedTankWalls);
    this.activeDispatch = device.createBuffer({
      label: "Uniform reference active indirect dispatches", size: activeRegionBytes,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    this.velocityExtrapolator = new WebGPUUniformVelocityExtrapolator(
      device, [nx, ny, nz], [
        scene.container.width_m / nx,
        scene.container.height_m / ny,
        scene.container.depth_m / nz,
      ], this.params, this.surfaceA, this.velocityD,
      this.velocityA, this.velocityC, this.transportA, this.transportB,
      this.activeRegion, this.activeRegionEnabled ? this.activeDispatch : undefined,
    );
    this.extrapolationActiveStateTexture = this.velocityExtrapolator.activeStateTexture;
    this.extrapolationActiveFrontPassCeiling = this.velocityExtrapolator.activeFrontPassCeiling;
    this.reductions = device.createBuffer({ label: "Uniform reference diagnostics and volume control", size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.conditioningScratch = device.createBuffer({ label: "Uniform reference compatibility scratch", size: allocation.conditioningBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
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

    this.mainLayout = device.createBindGroupLayout({ entries: [
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
    ] });
    this.pressureMultigrid = new WebGPUUniformPressureMultigrid(device, [nx, ny, nz], [
      scene.container.width_m / nx,
      scene.container.height_m / ny,
      scene.container.depth_m / nz,
    ], this.pressureSchedule, this.activeRegionEnabled ? this.activeDispatch : undefined);
    this.mainPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.mainLayout] });
    const sampler = device.createSampler({ minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    const group = (velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture,
      pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture,
      heightIn: GPUTexture, heightOut: GPUTexture, predicted = velocityIn,
      reversed = velocityIn, transport = this.transportA, surface = volumeIn,
      gammaRead = this.gammaA, gammaWrite = this.gammaB,
      boundaryRead = this.boundaryVelocityA, boundaryWrite = this.boundaryVelocityB,
      velocityPhase = volumeIn) => device.createBindGroup({
        layout: this.mainLayout, entries: [
          { binding: 0, resource: velocityIn.createView() }, { binding: 1, resource: velocityOut.createView() },
          { binding: 2, resource: pressureIn.createView() }, { binding: 3, resource: pressureOut.createView() },
          { binding: 4, resource: volumeIn.createView() }, { binding: 5, resource: volumeOut.createView() },
          { binding: 6, resource: { buffer: this.params } }, { binding: 7, resource: heightIn.createView() },
          { binding: 8, resource: heightOut.createView() }, { binding: 9, resource: { buffer: this.reductions } },
          { binding: 10, resource: { buffer: this.rigidSystem.stateBuffer } }, { binding: 11, resource: { buffer: this.rigidExchange } },
          { binding: 12, resource: predicted.createView() }, { binding: 13, resource: reversed.createView() },
          { binding: 14, resource: transport.createView() }, { binding: 15, resource: sampler },
          { binding: 16, resource: velocityPhase.createView() },
          { binding: 19, resource: { buffer: this.conditioningScratch } },
          { binding: 20, resource: surface.createView() }, { binding: 21, resource: this.terrainTexture.createView() },
          { binding: 24, resource: gammaRead.createView() }, { binding: 25, resource: gammaWrite.createView() },
          { binding: 26, resource: { buffer: boundaryRead } }, { binding: 27, resource: { buffer: boundaryWrite } },
          { binding: 28, resource: { buffer: this.macCormackAuditBinding } },
          { binding: 29, resource: { buffer: this.activeRegion } },
          { binding: 30, resource: { buffer: this.activeScratch } },
        ],
      });
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
    this.pressureMultigridGroup = group(this.velocityB, this.velocityA, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA, this.velocityB, this.velocityB, this.transportA, this.volumeB, this.gammaA, this.gammaB, this.boundaryVelocityB, this.boundaryVelocityA);
    this.projectGroup = group(this.velocityB, this.velocityA, this.pressureMultigrid.pressureTexture, this.pressureA, this.volumeB, this.volumeA, this.heightB, this.heightA, this.velocityB, this.velocityB, this.transportA, this.volumeB, this.gammaA, this.gammaB, this.boundaryVelocityB, this.boundaryVelocityA);
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
    const count = nx * ny * nz;
    this.info = {
      nx, ny, nz, storedNy: ny, cellCount: count, equivalentUniformCells: count,
      compressionRatio: 1, activeCompressionRatio: 1, activeSampleCount: count,
      regularLayers: ny, maximumNeighborDelta: 0, gridKind: "uniform",
      cellSize_m: Math.min(scene.container.width_m / nx, scene.container.height_m / ny, scene.container.depth_m / nz),
      pressureIterations: 0, pressureSolver: `CM11a dense LCP multigrid (${this.pressureSchedule.fullCycles} Full-Cycles + ${this.pressureSchedule.vCycles} V-Cycles, ${this.pressureSchedule.preSweeps}/${this.pressureSchedule.postSweeps} pre/post PRBGS)`,
      allocatedBytes: allocation.allocatedBytes + this.pressureMultigrid.allocatedBytes
        + activeRegionBytes * 3 + activeSummaryBytes + packedTankWalls.byteLength
        + (this.symmetryStageAuditMacCormackBuffer ? 0 : 16), quality,
      submittedTime_s: 0, simulatedTime_s: 0, completedTime_s: 0,
      simulationLag_s: 0, encodedSteps: 0, maximumTallCellHeight: 0,
      volumeControl: true,
      hostFluidAuthority: "gpu-resident", hostSimulationSizedWorkItems: 0,
      hostSchedulingUsesReadback: false,
    };
    this.volumeTexture = this.volumeA;
    this.columnBaseTexture = this.heightA;
    this.velocityTexture = this.velocityA;
    this.preProjectionVelocityTexture = this.velocityB;
    this.extrapolatedVelocityTexture = this.transportA;
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
    const shaderModule = compiler.createShaderModule({ label: "Uniform reference kernels", code: uniformReferenceComputeShader });
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
          uniformBindGroupLayout: this.mainLayout,
          shaderSource: `${uniformReferenceComputeShader}\n${this.pressureMultigrid.shaderFragment}`,
          signal,
        });
        this.publishUniformPipelineFacts();
      },
    });
    tasks.push({ id: "uniform.surface.initial", phase: "upload", label: "Reconstruct smooth t=0 surface", dependencies: [pipelineReadyId, extrapolationReadyId, multigridReadyId], run: () => { this.encodeInitialPresentationSurface(); } });
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
    this.runDirect(encoder, "Uniform initial Sec. 3.3 interface authority",
      this.pipelines.extrapolationAuthorityDense, this.extrapolationAuthorityGroup,
      [Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4)]);
    if (this.densityPostProcessing) {
      this.run(encoder, "Uniform initial post-process blur x", this.pipelines.postprocessBlurX, this.postprocessBlurXGroup);
      this.run(encoder, "Uniform initial post-process blur y", this.pipelines.postprocessBlurY, this.postprocessBlurYGroup);
      this.run(encoder, "Uniform initial post-process blur z", this.pipelines.postprocessBlurZ, this.postprocessBlurZGroup);
      this.run(encoder, "Uniform initial sub-grid surface resolve", this.pipelines.postprocessResolve, this.postprocessResolveGroup);
    } else if (sceneHasSphericalContainer(this.scene)) {
      // Curved vessels render the conserved density directly. A wall film is a
      // rectangular-tank presentation device and produces a false inner shell
      // when reconstructed against the analytic sphere.
      encoder.copyTextureToTexture({ texture: this.volumeA }, { texture: this.surfaceB },
        [this.info.nx, this.info.ny, this.info.nz]);
    } else {
      this.run(encoder, "Uniform initial wall-film resolve", this.pipelines.wallFilmResolve, this.wallFilmResolveGroup);
    }
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
      this.scene.fluid.density_kg_m3, this.scene.fluid.dynamicViscosity_Pa_s, 1, 0,
      this.scene.fluid.surfaceTension_N_m, c.fluidWallMode === "no-slip" ? 1 : 0, activeBodyCount, c.top === "open" ? 1 : 0,
      outlet?.x ?? 0, outlet?.y ?? 0, outlet?.z ?? 0, inflow?.radius_m ?? 0,
      inflow?.velocity_m_s.x ?? 0, inflow?.velocity_m_s.y ?? 0, inflow?.velocity_m_s.z ?? 0, this.inflowBoundary?.apertureScale ?? 0,
      inflowStrength, this.referenceVolumeCells, c.fillFraction * this.info.ny, 4,
      this.sharpeningStrength, this.sharpeningDistance,
      sceneHasSphericalContainer(this.scene) ? 1 : 0,
      c.depthBoundary === "symmetry" ? 1 : 0,
      drop?.centre_m.x ?? 0, drop?.centre_m.y ?? 0, drop?.centre_m.z ?? 0, drop?.radius_m ?? 0,
      drop?.halfHeight_m ?? 0, this.liquidOnlyVelocityAdvection ? 1 : 0,
      this.tankWallScratchOffsetWords, 0,
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
    const finite = (key: string, fallback: number, minimum: number, maximum: number) => {
      const value = Number(values[key]);
      return Number.isFinite(value) ? Math.min(maximum, Math.max(minimum, value)) : fallback;
    };
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
    this.sharpeningStrength = finite("sharpeningStrength", 1, 0.25, 2);
    this.sharpeningDistance = finite("sharpeningDistance", 2.1, 0.1, 3.1);
    this.solidExcessCorrection = values.solidExcessCorrection !== "off";
    this.rigidCoupling = values.rigidCoupling !== "off";
    this.paperTimeStep = values.timeStep !== "scene";
    this.velocityTransport = values.velocityTransport === "maccormack"
      ? "maccormack" : "semi-lagrangian";
    this.liquidOnlyVelocityAdvection = values.liquidOnlyVelocityAdvection === "on";
    if (refreshPresentation && this.pipelines) this.encodeInitialPresentationSurface();
  }

  private initializeVolumeAndTerrain(): void {
    const { nx, ny, nz } = this.info;
    const c = this.scene.container;
    const terrain = terrainColumnHeights(this.scene, nx, nz);
    const cellHeight = c.height_m / ny;
    const volume = new Float32Array(nx * ny * nz);
    const dam = sceneDamBreakBox(this.scene);
    let initial = 0;
    const wetMinimum = [nx, ny, nz];
    const wetMaximum = [0, 0, 0];
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const aboveGround = (y + 0.5) * cellHeight > terrain[x + nx * z];
      const containerOpen = sphericalContainerOpenFractionAtCell(this.scene, x, y, z, [nx, ny, nz]);
      const base = this.scene.fluid.initialCondition === "dam-break"
        ? damBreakBoxContains(dam, (x + 0.5) / nx, (y + 0.5) / ny, (z + 0.5) / nz)
        : (y + 0.5) / ny <= c.fillFraction;
      const liquidFraction = aboveGround
        ? initialLiquidFractionAtCell(this.scene, x, y, z, [nx, ny, nz], base) : 0;
      // Both fractions use the same eight sub-cell samples. Taking their
      // intersection as a minimum preserves a hemisphere that coincides with
      // the spherical vessel; multiplying would square its cut-cell fringe.
      const density = Math.min(containerOpen, liquidFraction);
      volume[x + nx * (y + ny * z)] = density;
      initial += density;
      if (density > 1e-5) {
        wetMinimum[0] = Math.min(wetMinimum[0]!, x);
        wetMinimum[1] = Math.min(wetMinimum[1]!, y);
        wetMinimum[2] = Math.min(wetMinimum[2]!, z);
        wetMaximum[0] = Math.max(wetMaximum[0]!, x + 1);
        wetMaximum[1] = Math.max(wetMaximum[1]!, y + 1);
        wetMaximum[2] = Math.max(wetMaximum[2]!, z + 1);
      }
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
    const words = new Uint32Array(UNIFORM_ACTIVE_LEVEL_BASE_WORD
      + UNIFORM_ACTIVE_MAX_LEVELS * UNIFORM_ACTIVE_LEVEL_WORDS);
    const empty = wetMaximum.every((value) => value === 0);
    const minimum = !this.activeRegionEnabled || empty ? [0, 0, 0]
      : wetMinimum.map((value) => Math.max(0, value - 8));
    const maximum = !this.activeRegionEnabled ? dimensions
      : empty ? [1, 1, 1] : wetMaximum.map((value, axis) =>
        Math.min(dimensions[axis]!, value + 8));
    // Words 0..5 retain the padded wet/source box from the previous step.
    // Words 7..12 are the union of two consecutive boxes, giving ping-pong
    // targets one clearing tail without retaining the entire swept history.
    words.set(minimum, 0);
    words.set(maximum, 3);
    words.set(minimum, 7);
    words.set(maximum, 10);
    words.set(maximum.map((value, axis) => Math.ceil((value - minimum[axis]!) / 4)), 13);
    this.pressureMultigrid.levelPhysicalDimensions.forEach((level, index) => {
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
  }

  private upload3DF32(texture: GPUTexture, values: Float32Array, nx: number, ny: number, nz: number): void {
    const rowBytes = nx * 4, padded = Math.ceil(rowBytes / 256) * 256;
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
    if (this.activeRegionEnabled) {
      pass.dispatchWorkgroupsIndirect(this.activeDispatch, UNIFORM_ACTIVE_MAIN_DISPATCH_OFFSET);
    } else pass.dispatchWorkgroups(
      Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
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

  private encodeActiveRegion(encoder: GPUCommandEncoder, externalSources: boolean): void {
    if (!this.pipelines) throw new Error("Uniform reference pipelines are not initialized");
    if (!this.activeRegionEnabled) return;
    if (externalSources) {
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
      (UNIFORM_ACTIVE_LEVEL_BASE_WORD + UNIFORM_ACTIVE_MAX_LEVELS * UNIFORM_ACTIVE_LEVEL_WORDS) * 4);
    encoder.copyBufferToBuffer(this.activeScratch, 0, this.activeDispatch, 0,
      (UNIFORM_ACTIVE_LEVEL_BASE_WORD + UNIFORM_ACTIVE_MAX_LEVELS * UNIFORM_ACTIVE_LEVEL_WORDS) * 4);
  }

  private encodeVelocityExtrapolation(
    encoder: GPUCommandEncoder,
    predicted: boolean,
    seam?: (phase: GPUTimestampPhase) => void,
  ): void {
    if (!this.pipelines) throw new Error("Uniform reference pipelines are not initialized");
    this.run(encoder, "Uniform Sec. 3.3 rho-prime and face authority",
      this.pipelines.extrapolationAuthority, this.extrapolationAuthorityGroup);
    if (!predicted && this.symmetryStageAuditTextures) encoder.copyTextureToTexture(
      { texture: this.surfaceA },
      { texture: this.symmetryStageAuditTextures.extrapolationDensityAuthority },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (!predicted) seam?.(UNIFORM_ADVANCE_PHASE.extensionAuthority);
    this.velocityExtrapolator.encode(encoder, predicted, !predicted && seam ? ((stage) => seam(
      stage === "narrow-band-front" ? UNIFORM_ADVANCE_PHASE.extensionFront
        : UNIFORM_ADVANCE_PHASE.extensionHierarchy,
    )) : undefined);
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
      extrapolationFrontSweeps: this.velocityExtrapolator.activeFrontPassCeiling,
      extrapolationHierarchyLevels: this.velocityExtrapolator.hierarchyLevelCount,
      extrapolationPassesPerInvocation: this.velocityExtrapolator.encodedPassCount,
      multigridLevels: this.pressureMultigrid.levelCount,
      multigridPasses,
      multigridPassesTotal: Object.values(multigridPasses).reduce((sum, count) => sum + count, 0),
      pressureSchedule: this.pressureSchedule,
    };
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
    this.encodeActiveRegion(encoder, strength > 0 || drop !== undefined);
    if (this.symmetryStageAuditTextures) encoder.copyTextureToTexture(
      { texture: this.velocityA }, { texture: this.symmetryStageAuditTextures.preExtrapolationVelocity },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (this.symmetryStageAuditTextures) encoder.copyTextureToTexture(
      { texture: this.volumeA }, { texture: this.symmetryStageAuditTextures.previousRawDensity },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (this.symmetryStageAuditNegativeBoundaryVelocity) encoder.copyBufferToBuffer(
      this.boundaryVelocityA, 0, this.symmetryStageAuditNegativeBoundaryVelocity, 0,
      this.negativeBoundaryVelocityBytes,
    );
    this.encodeVelocityExtrapolation(encoder, false, seam);

    // Sec. 3.6 must see the updated solid geometry before Sec. 3.4 excludes
    // solid donors. Without this pair, density in cells newly covered by a
    // moving body is skipped by beta construction and then overwritten with
    // zero by the gather, so the supposedly conservative operator loses the
    // displaced liquid before the historical post-sharpening cleanup runs.
    if (this.solidExcessCorrection && (activeBodies.length > 0 || sceneHasTerrain(this.scene)
      || sceneHasSphericalContainer(this.scene))) {
      encoder.clearBuffer(this.conditioningScratch);
      this.run(encoder, "Uniform moving-solid entry excess scatter",
        this.pipelines.scatterSolidExcess, this.solidEntryScatterGroup);
      this.run(encoder, "Uniform moving-solid entry excess resolve",
        this.pipelines.resolveSolidExcess, this.solidEntryResolveGroup);
    }

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
    if (this.symmetryStageAuditTextures) encoder.copyTextureToTexture(
      { texture: this.volumeB }, { texture: this.symmetryStageAuditTextures.densityAdvection },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (this.symmetryStageAuditTextures) encoder.copyTextureToTexture(
      { texture: this.gammaA }, { texture: this.symmetryStageAuditTextures.gammaPostAdvection },
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
      encoder.copyTextureToTexture({ texture: this.volumeA }, { texture: this.volumeB },
        [this.info.nx, this.info.ny, this.info.nz]);
      encoder.copyTextureToTexture({ texture: this.gammaB }, { texture: this.gammaA },
        [this.info.nx, this.info.ny, this.info.nz]);
    }
    if (this.symmetryStageAuditTextures) encoder.copyTextureToTexture(
      { texture: this.volumeB }, { texture: this.symmetryStageAuditTextures.densityDiffusion },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (this.symmetryStageAuditTextures) encoder.copyTextureToTexture(
      { texture: this.gammaA }, { texture: this.symmetryStageAuditTextures.gammaPostDiffusion },
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
        encoder.copyTextureToTexture(
          { texture: this.volumeA }, { texture: this.volumeB },
          [this.info.nx, this.info.ny, this.info.nz],
        );
        seam?.(UNIFORM_ADVANCE_PHASE.interfaceSharpening);
      }
    }
    if (this.symmetryStageAuditTextures) encoder.copyTextureToTexture(
      { texture: this.volumeB }, { texture: this.symmetryStageAuditTextures.densitySharpening },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    if (this.solidExcessCorrection && (activeBodies.length > 0 || sceneHasTerrain(this.scene)
      || sceneHasSphericalContainer(this.scene))) {
      encoder.clearBuffer(this.conditioningScratch);
      this.run(encoder, "Uniform partial-solid excess scatter", this.pipelines.scatterSolidExcess, this.solidExcessScatterGroup);
      this.run(encoder, "Uniform partial-solid excess resolve", this.pipelines.resolveSolidExcess, this.solidExcessResolveGroup);
      seam?.(UNIFORM_ADVANCE_PHASE.solidExcess);
    }
    encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
    // Algorithm 1 steps 3-4: advect/force velocity after the surface-density
    // update, then enforce incompressibility.
    if (this.velocityTransport === "maccormack") {
      // CM11b Sec. 3.5 modified MacCormack with local-extrema fallback. Extend
      // the forward prediction before the reverse trace so every lookup has a
      // defined velocity; body forces are applied exactly once in correction.
      this.run(encoder, "Uniform bounded MacCormack velocity prediction",
        this.pipelines.advect, this.advectGroup);
      if (this.symmetryStageAuditTextures) encoder.copyTextureToTexture(
        { texture: this.velocityC }, { texture: this.symmetryStageAuditTextures.velocityPrediction },
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
      if (this.symmetryStageAuditTextures) encoder.copyTextureToTexture(
        { texture: this.velocityB }, { texture: this.symmetryStageAuditTextures.velocityPrediction },
        [this.info.nx, this.info.ny, this.info.nz],
      );
      // The audit schema predates the selectable transport. Publish identity
      // placeholders for its MacCormack-only intermediate stages so a
      // semi-Lagrangian run never exposes stale texture contents as evidence.
      if (this.symmetryStageAuditTextures) {
        encoder.copyTextureToTexture(
          { texture: this.velocityB }, { texture: this.velocityD },
          [this.info.nx, this.info.ny, this.info.nz],
        );
        encoder.copyTextureToTexture(
          { texture: this.transportA }, { texture: this.transportB },
          [this.info.nx + 2, this.info.ny + 2, this.info.nz + 2],
        );
      }
    }
    if (this.symmetryStageAuditTextures) encoder.copyTextureToTexture(
      { texture: this.velocityB }, { texture: this.symmetryStageAuditTextures.velocityAdvection },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    seam?.(UNIFORM_ADVANCE_PHASE.advectionCorrection);
    this.pressureMultigrid.encode(encoder, this.pressureMultigridGroup,
      seam && ((stage) => seam(UNIFORM_PRESSURE_STAGE_PHASE[stage])));
    this.run(encoder, "Uniform pressure projection", this.pipelines.project, this.projectGroup);
    if (this.symmetryStageAuditTextures) encoder.copyTextureToTexture(
      { texture: this.velocityA }, { texture: this.symmetryStageAuditTextures.pressureProjection },
      [this.info.nx, this.info.ny, this.info.nz],
    );
    seam?.(UNIFORM_ADVANCE_PHASE.pressureProjection);
    if (this.rigidCoupling && activeBodies.length > 0) {
      this.run(encoder, "Uniform rigid-body coupling", this.pipelines.coupleRigid, this.rigidGroup);
      encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
      encoder.copyTextureToTexture({ texture: this.velocityB }, { texture: this.velocityA }, [this.info.nx, this.info.ny, this.info.nz]);
      encoder.copyBufferToBuffer(this.boundaryVelocityB, 0, this.boundaryVelocityA, 0, this.negativeBoundaryVelocityBytes);
      const cellVolume = c.width_m * c.height_m * c.depth_m / (this.info.nx * this.info.ny * this.info.nz);
      this.rigidSystem.encode(encoder, dt, cellVolume, 1, c.height_m / this.info.ny);
      seam?.(UNIFORM_ADVANCE_PHASE.rigidCoupling);
    }
    // Sec. 3.8 remains the optional global reconstruction. Box tanks may expose
    // mass-proportional wall films; a spherical vessel deliberately renders
    // the conserved field directly so it never gains a false inner shell.
    if (this.densityPostProcessing) {
      this.run(encoder, "Uniform post-process blur x", this.pipelines.postprocessBlurX, this.postprocessBlurXGroup);
      this.run(encoder, "Uniform post-process blur y", this.pipelines.postprocessBlurY, this.postprocessBlurYGroup);
      this.run(encoder, "Uniform post-process blur z", this.pipelines.postprocessBlurZ, this.postprocessBlurZGroup);
      this.run(encoder, "Uniform sub-grid surface resolve", this.pipelines.postprocessResolve, this.postprocessResolveGroup);
    } else if (sceneHasSphericalContainer(this.scene)) {
      encoder.copyTextureToTexture({ texture: this.volumeA }, { texture: this.surfaceB },
        [this.info.nx, this.info.ny, this.info.nz]);
    } else {
      this.run(encoder, "Uniform wall-film resolve", this.pipelines.wallFilmResolve, this.wallFilmResolveGroup);
    }
    seam?.(UNIFORM_ADVANCE_PHASE.densityPostProcess);
    // The final phase closes on the reduction pass itself (its end-of-pass
    // counter) rather than on a synthetic marker pass after it: a marker
    // touches no frame resource, so Metal is free to schedule it early and its
    // timestamp lands before the boundary it is meant to close.
    physicsTrace?.completeFinalPhaseOnNextPass(UNIFORM_ADVANCE_PHASE.diagnosticsReduction);
    this.run(encoder, "Uniform diagnostics reduction", this.pipelines.reduce, this.reductionGroup);
    physicsCPUTrace?.completePhase(UNIFORM_ADVANCE_PHASE.diagnosticsReduction);
    physicsTrace?.resolve(encoder);
    physicsQueueTrace?.begin();
    this.device.queue.submit([encoder.finish()]);
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
    const submittedTime = this.lastTime;
    void this.device.queue.onSubmittedWorkDone().then(() => {
      if (!this.disposed) this.info.completedTime_s = Math.max(this.info.completedTime_s ?? 0, submittedTime);
    }).catch(() => {});
    return true;
  }

  async readStats(): Promise<GPUEulerianInfo> {
    if (this.disposed || this.readbackPending) return this.info;
    this.readbackPending = true;
    this.statsReadback ??= this.device.createBuffer({ label: "Uniform reference diagnostics readback", size: 176, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Uniform reference diagnostics readback" });
    encoder.copyBufferToBuffer(this.reductions, 0, this.statsReadback, 0, 20);
    encoder.copyBufferToBuffer(this.pressureMultigrid.diagnostics, 0, this.statsReadback, 32, 60);
    encoder.copyBufferToBuffer(this.velocityExtrapolator.convergenceDiagnostics, 0, this.statsReadback, 96, 16);
    encoder.copyBufferToBuffer(this.activeRegion, 0, this.statsReadback, 112, 64);
    this.device.queue.submit([encoder.finish()]);
    try {
      await this.statsReadback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(this.statsReadback.getMappedRange().slice(0));
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
        uniformCM11aCoarseWorstRow: words[22] & 0xffff,
        uniformCM11aCoarseWorstRowActive: (words[22] & 0x1_0000) !== 0,
        uniformCM11aCoarseWorstRowHalo: (words[22] & 0x2_0000) !== 0,
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
      });
      return this.info;
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

  applySceneUniforms(scene: SceneDescription): void {
    this.scene = scene;
    this.device.queue.writeBuffer(this.activeScratch, this.tankWallScratchOffsetWords * 4,
      packTankWallField(scene.container.wallField));
    this.inflowBoundary = scene.fluid.inflow
      ? createInflowGridBoundary(scene.fluid.inflow, scene.container, [this.info.nx, this.info.ny, this.info.nz])
      : undefined;
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
      ...Object.values(this.symmetryStageAuditTextures ?? {}),
      this.heightA, this.heightB, this.terrainTexture,
      this.transportA, this.transportB,
    ])) texture.destroy();
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
    this.rigidSystem.destroy();
    this.rigidExchange.destroy();
    this.statsReadback?.destroy();
  }
}

/**
 * The uniform method's pipeline graph, declared beside the encoder whose seams
 * it names. Every `phaseLabels` entry below is a `UNIFORM_ADVANCE_PHASE` label
 * — the invariant tests hold the two to an exact one-to-one partition, so a
 * stage card's figure is always the hardware time of its own passes.
 */

const uniformFacts = (context: FluidPipelineContext) => context.info?.uniformPipelineFacts;

/**
 * The one gate the uniform method exposes, resolved exactly as
 * `uniformReferenceSolverOptions` resolves it: "scene" means Sec. 3.8 runs
 * only where the presentation depends on it. Shared so the pipeline panel's
 * lamp and the constructed solver can never disagree.
 */
export function uniformDensityPostProcessingEnabled(
  densityPostProcessing: unknown,
  sceneId: string | undefined,
): boolean {
  if (densityPostProcessing === "on") return true;
  if (densityPostProcessing !== "scene") return false;
  return false;
}

const uniformExtensionChip = (context: FluidPipelineContext): string => {
  const facts = uniformFacts(context);
  return facts
    ? `${facts.extrapolationPassesPerInvocation} passes · ${facts.extrapolationHierarchyLevels} MIP levels`
    : "narrow-band FIM + hierarchy fill";
};

const uniformExtensionTip = () => ({
  summary: "Sec. 3.3: builds the interface authority field, marches extended face velocities across the narrow band with a fast-iterative-method front, then fills the far field through a coarse hierarchy so every advection lookup lands on defined velocity.",
  reads: "MAC velocity, surface density",
  writes: "extended MAC velocity, FIM active state",
  feeds: "conservative density and velocity advection",
} as const);

const UNIFORM_FLUID_STAGES: readonly FluidPipelineStage[] = [
  {
    id: "velocity-extension",
    band: "extension",
    side: "left",
    label: "Velocity extension",
    phaseLabels: [
      UNIFORM_ADVANCE_PHASE.extensionAuthority.label,
      UNIFORM_ADVANCE_PHASE.extensionFront.label,
      UNIFORM_ADVANCE_PHASE.extensionHierarchy.label,
    ],
    tip: uniformExtensionTip(),
    controls: [
      {
        kind: "param-choice",
        param: "activeRegion",
        label: "Dispatch",
        options: [
          { value: "on", label: "Active region" },
          { value: "off", label: "Dense control" },
        ],
      },
      {
        kind: "readout",
        label: "Work box",
        hint: "Latest GPU-measured dispatch volume as a share of the uniform lattice.",
        value: (context) => context.values.activeRegion === "off"
          ? "100% dense"
          : context.info?.uniformActiveRegionFraction !== undefined
            ? `${(100 * context.info.uniformActiveRegionFraction).toFixed(1)}%`
            : "—",
      },
      {
        kind: "readout",
        label: "Front sweeps",
        hint: "Bounded FIM iteration ceiling; indirect work resolves early and later sweeps are no-ops on converged cells.",
        value: (context) => {
          const facts = uniformFacts(context);
          return facts ? `${facts.extrapolationFrontSweeps}` : "—";
        },
      },
    ],
    state: () => "on",
    chip: uniformExtensionChip,
  },
  {
    id: "density-advection",
    band: "surface",
    side: "left",
    label: "Density advection",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.densityAdvection.label],
    tip: {
      summary: "Sec. 3.4: the modified conservative semi-Lagrangian operator — trace gamma and beta backward along the extended velocity, scatter density deficits, gather the conserved result. Mass is redistributed, never created.",
      reads: "surface density, extended MAC velocity",
      writes: "surface density, gamma",
      feeds: "gamma diffusion",
    },
    state: () => "on",
    chip: () => "3 passes · mass-conserving",
  },
  {
    id: "gamma-diffusion",
    band: "surface",
    side: "right",
    label: "Gamma diffusion",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.gammaDiffusion.label],
    tip: {
      summary: "Sec. 3.4: each axis gathers neighbouring half-fluxes from one Jacobi snapshot while transferring the matching donor density; completed axes feed the next axis.",
      reads: "surface density, gamma",
      writes: "surface density, gamma",
      feeds: "interface sharpening",
    },
    toggle: {
      param: "gammaDiffusion", on: "on", off: "off",
      hint: "Toggle Sec. 3.4 axis-Jacobi gamma diffusion. Density transport remains complete when it is off.",
    },
    controls: [{
      kind: "param-range",
      param: "gammaDiffusionIterations",
      label: "Iterations",
      min: 1, max: 7, step: 1,
      hint: "One iteration is three snapshot axis passes; the paper permits one through seven.",
      enabled: (context) => context.values.gammaDiffusion !== "off",
    }],
    state: (context) => context.values.gammaDiffusion === "off" ? "off" : "on",
    chip: (context) => context.values.gammaDiffusion === "off"
      ? "off · identity handoff"
      : `${3 * Number(context.values.gammaDiffusionIterations ?? UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS)} passes · ${context.values.gammaDiffusionIterations ?? UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS} iterations`,
  },
  {
    id: "interface-sharpening",
    band: "surface",
    side: "left",
    label: "Density correction",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.interfaceSharpening.label],
    tip: {
      summary: "Sec. 3.5: computes and applies the local density correction that steepens the smeared liquid-air interface. Its strength scales the paper's pseudo-time dose; local mass return is exposed as the following stage.",
      reads: "surface density, gamma",
      writes: "surface density",
      feeds: "local mass return, then velocity prediction as the liquid mask",
    },
    toggle: {
      param: "densitySharpening", on: "on", off: "off",
      hint: "Toggle the Sec. 3.5 interface-sharpening correction and its dependent mass-return stage.",
    },
    controls: [{
      kind: "param-range",
      param: "sharpeningStrength",
      label: "Strength",
      unit: "×",
      min: 0.25, max: 2, step: 0.05, digits: 2,
      hint: "Multiplier over the paper's 3dt sharpening pseudo-time dose.",
      enabled: (context) => context.values.densitySharpening !== "off",
    }],
    state: (context) => context.values.densitySharpening === "off" ? "off" : "on",
    chip: (context) => context.values.densitySharpening === "off"
      ? "off · advected density"
      : `1 pass · ${Number(context.values.sharpeningStrength ?? 1).toFixed(2)}× dose`,
  },
  {
    id: "sharpening-mass-correction",
    band: "surface",
    side: "right",
    label: "Local mass return",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.sharpeningMassCorrection.label],
    tip: {
      summary: "Sec. 3.5 Algorithm 2: traces each removed density parcel toward the 0.5 iso-contour, scatters it locally, and resolves the deposits. Disabling it intentionally exposes the raw, non-conservative density correction while preserving a valid downstream field.",
      reads: "corrected density, per-cell removed mass",
      writes: "mass-returned surface density",
      feeds: "partial-solid excess and pressure classification",
      gate: "interface sharpening and local mass return are both on",
    },
    toggle: {
      param: "sharpeningMassCorrection", on: "on", off: "off",
      hint: "Toggle only Algorithm 2's local conservation step; density correction remains active.",
    },
    controls: [{
      kind: "param-range",
      param: "sharpeningDistance",
      label: "Trace distance",
      unit: "cells",
      min: 0.1, max: 3.1, step: 0.1, digits: 1,
      hint: "Maximum gradient-trace distance D; the paper explores 1.1–3.1 cells, and values below that keep returned mass local.",
      enabled: (context) => context.values.densitySharpening !== "off"
        && context.values.sharpeningMassCorrection !== "off",
    }],
    state: (context) => context.values.densitySharpening === "off"
      ? "unavailable"
      : context.values.sharpeningMassCorrection === "off" ? "off" : "on",
    chip: (context) => context.values.densitySharpening === "off"
      ? "requires density correction"
      : context.values.sharpeningMassCorrection === "off"
        ? "off · non-conservative ablation"
        : `2 passes · D ${Number(context.values.sharpeningDistance ?? 2.1).toFixed(1)} cells`,
  },
  {
    id: "solid-excess",
    band: "surface",
    side: "right",
    label: "Partial-solid excess",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.solidExcess.label],
    tip: {
      summary: "Sec. 3.6: current density is reconciled before transport can mask newly covered donors, then checked again after sharpening. Excess beyond a cut cell's open fraction is scattered to open neighbours and resolved conservatively; genuinely enclosed excess is published as telemetry.",
      reads: "surface density, solid fractions",
      writes: "surface density",
      gate: "the scene has rigid bodies or terrain",
    },
    toggle: {
      param: "solidExcessCorrection", on: "on", off: "off",
      hint: "Toggle Sec. 3.6 cut-cell excess redistribution in scenes that contain solids.",
    },
    state: (context) => context.bodyCount > 0 || context.hasTerrain
      ? context.values.solidExcessCorrection === "off" ? "off" : "on"
      : "unavailable",
    chip: (context) => context.bodyCount > 0 || context.hasTerrain
      ? context.values.solidExcessCorrection === "off" ? "off · excess retained" : "4 passes · entry + post-sharpening"
      : "no solids in scene",
  },
  {
    id: "velocity-advection",
    band: "momentum",
    side: "left",
    label: "Velocity advection",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.advectionCorrection.label],
    tip: {
      summary: "Algorithm 1 step 3: configurable semi-Lagrangian or CM11b bounded MacCormack velocity transport, followed by gravity, viscosity, and surface tension.",
      reads: "extended MAC velocity; predicted and reverse fields in MacCormack mode",
      writes: "advected MAC velocity",
      feeds: "pressure solve",
    },
    controls: [{
      kind: "param-choice",
      param: "velocityTransport",
      label: "Transport",
      hint: "Choose the one-pass semi-Lagrangian update or the higher-order bounded MacCormack sequence.",
      options: [
        { value: "semi-lagrangian", label: "Semi-Lagrangian" },
        { value: "maccormack", label: "Bounded MacCormack" },
      ],
    }],
    state: () => "on",
    chip: (context) => context.values.velocityTransport === "maccormack"
      ? "forward + reverse + bounded correction"
      : "one backward-trace pass",
  },
  {
    id: "pressure-system",
    band: "pressure",
    side: "left",
    label: "System build",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.pressureSetup.label],
    tip: {
      summary: "CM11a setup: classify cell topology, build the divergence right-hand side from the advected velocity, and restrict both down the multigrid pyramid before any cycle runs.",
      reads: "advected MAC velocity, surface density, solid fractions",
      writes: "per-level topology + RHS",
      feeds: "multigrid cycles",
    },
    state: () => "on",
    chip: (context) => {
      const facts = uniformFacts(context);
      return facts
        ? `${facts.multigridPasses.setup} passes · ${facts.multigridLevels} levels`
        : "topology + RHS pyramid";
    },
  },
  {
    id: "pressure-cycles",
    band: "pressure",
    side: "right",
    label: "Multigrid cycles",
    phaseLabels: [
      UNIFORM_ADVANCE_PHASE.pressureFullCycles.label,
      UNIFORM_ADVANCE_PHASE.pressureVCycles.label,
    ],
    tip: {
      summary: "The CM11a LCP multigrid solve: full cycles first (coarsest-up, seeding every level), then V-cycles to polish. Each level runs projected red-black Gauss-Seidel sweeps with the liquid-air complementarity condition enforced per sweep.",
      reads: "per-level topology + RHS",
      writes: "pressure",
      feeds: "parity copy + fine residual",
    },
    controls: [
      {
        kind: "param-range",
        param: "pressureFullCycles",
        label: "Full-Cycles",
        min: 0, max: 5, step: 1,
        hint: "Coarsest-up CM11a Full-Cycles; the paper schedule uses three. Changing it rebuilds the precomputed pressure plan and resets to t=0.",
      },
      {
        kind: "param-range",
        param: "pressureVCycles",
        label: "V-Cycles",
        min: 0, max: 8, step: 1,
        hint: "Refinement V-Cycles after the Full-Cycles; the paper schedule uses four. Changing it rebuilds the precomputed pressure plan and resets to t=0.",
      },
      {
        kind: "param-range",
        param: "pressureSweeps",
        label: "Pre/post sweeps",
        min: 1, max: 8, step: 1,
        hint: "Projected red-black Gauss-Seidel sweeps before and after each coarse correction. Changing it rebuilds the precomputed pressure plan and resets to t=0.",
      },
      {
        kind: "readout",
        label: "Residual ∞-norm",
        hint: "Fine-level residual after the configured schedule, from diagnostics readback.",
        value: (context) => {
          const residual = (context.info as unknown as {
            uniformCM11aFineResidualInfinity?: number;
          } | null)?.uniformCM11aFineResidualInfinity;
          return residual === undefined || !Number.isFinite(residual)
            ? "—"
            : residual.toExponential(2);
        },
      },
    ],
    state: () => "on",
    chip: (context) => {
      const facts = uniformFacts(context);
      const configured = facts?.pressureSchedule;
      const fullCycles = configured?.fullCycles ?? Number(context.values.pressureFullCycles ?? UNIFORM_CM11A_FULL_CYCLES);
      const vCycles = configured?.vCycles ?? Number(context.values.pressureVCycles ?? UNIFORM_CM11A_V_CYCLES);
      const preSweeps = configured?.preSweeps ?? Number(context.values.pressureSweeps ?? UNIFORM_CM11A_PRE_SWEEPS);
      const postSweeps = configured?.postSweeps ?? Number(context.values.pressureSweeps ?? UNIFORM_CM11A_POST_SWEEPS);
      const schedule = `${fullCycles} full + ${vCycles} V · ${preSweeps}+${postSweeps} sweeps`;
      return facts
        ? `${facts.multigridPasses["full-cycle"] + facts.multigridPasses["v-cycle"]} passes · ${schedule}`
        : schedule;
    },
  },
  {
    id: "pressure-finish",
    band: "pressure",
    side: "left",
    label: "Solve finish",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.pressureFinish.label],
    tip: {
      summary: "Copies the converged pressure to the parity texture the projection reads and computes the fine-level residual the diagnostics report.",
      reads: "pressure",
      writes: "pressure (parity), residual diagnostics",
      feeds: "pressure projection",
    },
    state: () => "on",
    chip: (context) => {
      const facts = uniformFacts(context);
      return facts ? `${facts.multigridPasses.finish} passes` : "parity copy + residual";
    },
  },
  {
    id: "pressure-projection",
    band: "pressure",
    side: "right",
    label: "Pressure projection",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.pressureProjection.label],
    tip: {
      summary: "Subtracts the pressure gradient from the advected velocity, restoring a divergence-free field at every liquid face.",
      reads: "advected MAC velocity, pressure",
      writes: "divergence-free MAC velocity",
      feeds: "rigid coupling, next advance",
    },
    state: () => "on",
    chip: () => "1 pass",
  },
  {
    id: "rigid-coupling",
    band: "coupling",
    side: "left",
    label: "Rigid coupling",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.rigidCoupling.label],
    tip: {
      summary: "Two-way momentum exchange: the fluid pushes on each body through sampled pressure and drag, bodies push back on the face velocities, then the rigid system integrates poses for the next advance.",
      reads: "divergence-free MAC velocity, surface density, body poses",
      writes: "MAC velocity, body poses + momenta",
      gate: "the scene has rigid bodies",
    },
    toggle: {
      param: "rigidCoupling", on: "on", off: "off",
      hint: "Toggle two-way fluid/body momentum exchange and rigid integration; bodies remain solid pressure boundaries.",
    },
    state: (context) => context.bodyCount > 0
      ? context.values.rigidCoupling === "off" ? "off" : "on"
      : "unavailable",
    chip: (context) => context.bodyCount > 0
      ? context.values.rigidCoupling === "off"
        ? `off · ${context.bodyCount} ${context.bodyCount === 1 ? "body" : "bodies"} held`
        : `${context.bodyCount} ${context.bodyCount === 1 ? "body" : "bodies"}`
      : "no rigid bodies",
  },
  {
    id: "density-post-process",
    band: "output",
    side: "left",
    label: "Render density",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.densityPostProcess.label],
    tip: {
      summary: "Always reconstructs sub-half-cell liquid supported by tank walls and embedded solids at a mass-proportional thickness. Sec. 3.8 optionally adds its global gamma-blur reconstruction. The result never feeds simulation state.",
      reads: "surface density, gamma",
      writes: "render surface texture",
      gate: "wall films always; global Sec. 3.8 reconstruction when enabled",
    },
    controls: [{
      kind: "param-choice",
      param: "densityPostProcessing",
      label: "Sec. 3.8 reconstruction",
      hint: "Render-only surface smoothing; simulation state is identical either way. Changing it rebuilds the solver.",
      options: [
        { value: "scene", label: "Scene", hint: "On for symmetry and mini-dam scenes where sub-grid sheets are presentation-critical." },
        { value: "off", label: "Wall films", hint: "Mass-proportional solid-supported sheets only." },
        { value: "on", label: "Wall films + Sec. 3.8", hint: "Adds gamma blur and global sub-grid resolve." },
      ],
    }],
    toggle: {
      param: "densityPostProcessing", on: "on", off: "off",
      hint: "Toggle the render-only Sec. 3.8 density reconstruction without changing simulation physics.",
    },
    state: () => "on",
    chip: (context) => uniformDensityPostProcessingEnabled(
      context.values.densityPostProcessing, context.sceneId)
      ? "4 passes · Sec. 3.8 + wall films"
      : context.values.densityPostProcessing === "scene"
        ? "1 pass · wall films"
        : "1 pass · wall films",
  },
  {
    id: "diagnostics-reduction",
    band: "output",
    side: "right",
    label: "Diagnostics reduction",
    phaseLabels: [UNIFORM_ADVANCE_PHASE.diagnosticsReduction.label],
    tip: {
      summary: "Reduces liquid volume, front position, and maximum speed into the telemetry buffer the stats readback maps; the panel's drift and speed figures come from here.",
      reads: "surface density, MAC velocity",
      writes: "reductions buffer",
      feeds: "readStats() telemetry",
    },
    state: () => "on",
    chip: () => "1 pass",
  },
];

export const UNIFORM_FLUID_PIPELINE: FluidPipelineGraph = Object.freeze({
  methodId: "uniform",
  bands: [
    { id: "extension", label: "Velocity extension · Sec. 3.3" },
    { id: "surface", label: "Surface density · Secs. 3.4–3.6" },
    { id: "momentum", label: "Momentum transport" },
    { id: "pressure", label: "Pressure · CM11a multigrid" },
    { id: "coupling", label: "Rigid coupling" },
    { id: "output", label: "Output + diagnostics" },
  ],
  stages: UNIFORM_FLUID_STAGES,
});
