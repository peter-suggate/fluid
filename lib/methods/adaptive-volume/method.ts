import { type SparseCM12CorrectionControls } from "./correction-controls";
import { pressureCaptureParam, pressureCaptureDescriptor, SPARSE_CM12_PRESSURE_JOURNAL_SNAPSHOTS } from "./features/pressure-inspection/definition";
import { ALGORITHM_PARAMS } from "./features/algorithms/definition";
import { resolveMethodComposition } from "./composition";
import { SPARSE_CM12_ACTIVITY_POLICY } from "./features/adaptivity/policy";
import { ADAPTIVITY_PARAMS } from "./features/adaptivity/definition";
import type {
  MethodParamSpec,
  MethodParamValues,
  SimulationMethod,
} from "../../core/method-contract";
import { SPARSE_CM12_LENSES } from "./sparse-cm12-stage-lenses";
import { CM12_PAPER_DT_S } from "../../core/cm12-numerics";
import { adaptiveMassDiagnosticRows } from "./adaptive-mass-diagnostics";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from "./adaptive-mass-frame-pipeline";
import { sparseCM12PressureIterations, sparseCM12PressureRelativeTolerance, sparseCM12SharpeningDistance, sparseCM12SharpeningStrength, sparseCM12SharpeningTraceSteps, SPARSE_CM12_PRESSURE_ITERATIONS, SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE, SPARSE_CM12_SHARPENING_DISTANCE_CELLS, SPARSE_CM12_SHARPENING_STRENGTH, SPARSE_CM12_SHARPENING_TRACE_STEPS } from "./webgpu-sparse-cm12-resident";
import { sparseCM12ActivityPolicy, type SparseCM12ActivityPolicy } from "./features/adaptivity/policy";
import { WebGPUAdaptiveMassSolver } from "./webgpu-adaptive-mass-solver";
import type {
  SparseBrickResolution,
  SparseBrickFineResolution,
} from "./sparse-brick-atlas";
import {
  physicsExecutionBackendParams,
  resolvePhysicsExecutionBackend,
} from "../../core/physics-execution-backend";
import { RustFluid3DGPUSolverAdapter } from "../../physics-wasm/fluid3d-gpu-adapter";

/** Sparse-resolution controls consumed by the interactive solver factory. */
export type AdaptiveMassResolutionMode = "adaptive";

export interface AdaptiveMassSolverOptions extends SparseCM12CorrectionControls {
  /** Optional compatibility spelling; adaptive is the only production policy. */
  readonly resolutionMode?: AdaptiveMassResolutionMode;
  /** Test-only construction seam for manufacturing a fine-start transition. */
  readonly initialResolutionForQA?: SparseBrickResolution;
  /** Diagnostic fixed-domain setup: activate the initial atlas, including dry
   * support, without changing its cell sizes or advancing simulation time. */
  readonly initialAtlasResidentForQA?: boolean;
  /** Construction-time complete dyadic ladder maximum. Defaults to 8. */
  readonly brickFineResolution?: SparseBrickFineResolution;
  /** Renderer-facing samples per presentation-page edge. Defaults to the brick maximum. */
  readonly presentationPageResolution?: SparseBrickFineResolution;
  readonly surfaceMeshRefinement?: 1 | 2 | 4;
  /** Optional positive-power-of-two cap on hierarchical macro-leaf span. */
  readonly maximumMacroSpanBricks?: number;
  /** Physical world-growth page budget. Authored re-rung already owns complete
   * template topology and does not consume these page identities. */
  readonly topologyPageBudget?: number;
  /** Optional aggregate byte cap for simultaneous accepted/candidate resident
   * generations and transfer preparation. Omitted: derive from allowed growth. */
  readonly topologyGenerationMaximumBytes?: number;
  readonly surfaceFineRings?: number;
  readonly activityPolicy?: SparseCM12ActivityPolicy;
  /** Omitted only by direct diagnostic constructors, which retain scene-step behavior. */
  readonly timeStep?: "paper" | "scene";
  /** CM12 Algorithm 2's D, in finest cells. Omitted constructors use the shared 2.1 default. */
  readonly sharpeningDistance?: number;
  /** Forward-Euler substeps TraceAlongField may spend reaching D. */
  readonly sharpeningTraceSteps?: number;
  /** Multiplier of CM12 Algorithm 2's per-step removed-density dose. */
  readonly sharpeningStrength?: number;
  /** Whether Sec. 3.4 gamma diffusion runs. */
  readonly gammaDiffusionEnabled?: boolean;
  /** Whether Sec. 3.5's conservative surface-sharpening transform runs. */
  readonly surfaceSharpeningEnabled?: boolean;
  /** Validated column-height presentation policy; defaults to adaptive-coarse auto. */
  readonly presentationColumnHeightMode?: "off" | "auto" | "on";
  /** Shared reconstructed-distance presentation is the production default. */
  readonly presentationSurfaceMode?: "rdf" | "plic";
  /** @deprecated Use presentationColumnHeightMode; an explicit mode takes precedence. */
  readonly presentationColumnHeightEnabled?: boolean;
  /** Maximum one-reduction sparse MGPCG iterations encoded for each pressure solve. */
  readonly pressureIterations?: number;
  /** Relative L2 residual that stops further PCG arithmetic; zero runs the full budget. */
  readonly pressureRelativeTolerance?: number;
  /**
   * Reserve the pressure journal, so the pressure lab can capture a solve.
   *
   * A construction-time capability rather than a runtime toggle: the journal is
   * a tail range of the resident state buffer, whose size is fixed when the
   * solver is built. Off by default because the snapshot region scales with the
   * cell count, and a lane that never opens the lab must not pay for it.
   */
  readonly pressureJournal?: boolean;
}

const params: MethodParamSpec[] = [
  ...physicsExecutionBackendParams(true),
  ...ALGORITHM_PARAMS,
  ...ADAPTIVITY_PARAMS,

  {
    kind: "select",
    key: "presentationSurface",
    label: "Surface reconstruction",
    default: "rdf",
    tier: "coarse",
    update: "runtime",
    options: [{ value: "rdf", label: "Shared RDF" },
      { value: "plic", label: "Legacy PLIC field" }],
    hint: "Shared RDF reconstructs one watertight presentation field from accepted VOF fractions and PLIC normals. Legacy PLIC keeps the previous plane-support view. Both are presentation-only; transport remains volume-correct PLIC.",
  },

  {
    kind: "select",
    key: "presentationColumnHeight",
    label: "Column height",
    default: "auto",
    tier: "coarse",
    update: "runtime",
    options: [{ value: "auto", label: "Auto · coarse columns" },
      { value: "on", label: "On · local columns" },
      { value: "off", label: "Off · interface surface" }],
    hint: "Auto uses liquid volume to set surface height only in coarse columns filled continuously from the physical floor to a single surface. On uses a broader local check, including fine columns and some floating liquid. Off uses the interface surface throughout. Applies on the next simulation step without resetting the scene.",
  },
  {
    kind: "select",
    key: "surfaceMeshRefinement",
    label: "Surface mesh refinement",
    default: "2",
    tier: "coarse",
    update: "runtime",
    options: [{ value: "1", label: "×1 per surface cell" },
      { value: "2", label: "×2 per surface cell" },
      { value: "4", label: "×4 per surface cell" }],
    hint: "Target mesh spacing is the accepted cell width divided by this ratio. Shared boundaries retain their contour; unresolved geometry keeps finer triangles.",
  },
  {
    kind: "number",
    key: "pressureIterations",
    label: "Pressure iteration budget",
    default: SPARSE_CM12_PRESSURE_ITERATIONS,
    tier: "coarse",
    unit: "iterations",
    min: 8,
    max: 256,
    step: 8,
    digits: 0,
    update: "runtime",
    hint: "Maximum sparse Jacobi-PCG iterations per pressure solve. Lower budgets trade incompressibility for frame time; the device stops later blocks once the residual target is met.",
  },
  {
    kind: "number",
    key: "pressureRelativeTolerance",
    label: "Pressure early-stop tolerance",
    default: SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE,
    tier: "fine",
    unit: "rel. L2",
    min: 0,
    max: 1,
    step: 0.001,
    digits: 3,
    update: "runtime",
    hint: "Tests a fresh relative residual after each eight-iteration block and skips arithmetic in later fixed dispatches once it is met. Zero preserves fixed-budget execution; values through 1 are available for experimentation.",
  },

];

/**
 * The controls a live Sparse Geometric solver adopts.
 *
 * Clock and GPU candidate-policy values enter the
 * next advance through a small uniform. The per-frame preparation budget is a
 * GPU scheduler limit, not a host worklist: `advanceTo` still encodes the same
 * fixed/indirect dispatch sequence without reading queue state back.
 */
export const ADAPTIVE_MASS_RUNTIME_PARAM_KEYS = Object.freeze(
  params.filter(param => param.update === "runtime").map(param => param.key),
);

const boundedInteger = (value: unknown, fallback: number, minimum: number, maximum: number) =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(maximum, Math.max(minimum, Math.round(value))) : fallback;

const brickFineResolution = (value: unknown): SparseBrickFineResolution =>
  value === 4 || value === "4" ? 4 : value === 16 || value === "16" ? 16 : 8;

const presentationPageResolution = (
  _value: unknown,
  maximum: SparseBrickFineResolution,
): SparseBrickFineResolution => maximum;

const maximumMacroSpanBricks = (value: unknown): number | undefined => {
  if (value === undefined || value === "auto") return undefined;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isSafeInteger(parsed) && parsed >= 1
    && Number.isInteger(Math.log2(parsed)) ? parsed : undefined;
};

const selectorMode = (value: unknown): "surface" | "activity" | "coarse-first" =>
  value === "surface" ? "surface" : value === "activity" ? "activity" : "coarse-first";

const activityPolicy = (values: MethodParamValues): SparseCM12ActivityPolicy =>
  sparseCM12ActivityPolicy({
    ...values,
    activitySignals: selectorMode(values.selectorMode) !== "surface",
    coarseFirst: selectorMode(values.selectorMode) === "coarse-first",
  });

export function adaptiveMassSolverOptions(
  values: MethodParamValues,
): AdaptiveMassSolverOptions {
  resolveMethodComposition(values);
  const fineResolution = resolvePhysicsExecutionBackend(values) === "cpu"
    ? 8 : brickFineResolution(values.brickFineResolution);
  return {
    brickFineResolution: fineResolution,
    surfaceMeshRefinement: Number(values.surfaceMeshRefinement) === 1 ? 1
      : Number(values.surfaceMeshRefinement) === 4 ? 4 : 2,
    presentationPageResolution:
      presentationPageResolution(values.presentationPageResolution, fineResolution),
    maximumMacroSpanBricks: maximumMacroSpanBricks(values.maximumMacroSpanBricks),
    surfaceFineRings: boundedInteger(values.surfaceFineRings, 1, 1, 8),
    activityPolicy: activityPolicy(values),
    timeStep: values.timeStep === "scene" ? "scene" : "paper",
    gammaDiffusionEnabled: false,
    surfaceSharpeningEnabled: values.surfaceSharpening !== "off",
    densityCapacityRepairEnabled: false,
    volumeCorrectionEnabled: false,
    presentationColumnHeightMode: values.presentationColumnHeight === "off" ? "off"
      : values.presentationColumnHeight === "on" ? "on" : "auto",
    presentationSurfaceMode: values.presentationSurface === "plic" ? "plic" : "rdf",
    pressureIterations: sparseCM12PressureIterations(values.pressureIterations),
    pressureRelativeTolerance:
      sparseCM12PressureRelativeTolerance(values.pressureRelativeTolerance),
    sharpeningDistance: sparseCM12SharpeningDistance(values.sharpeningDistance),
    sharpeningTraceSteps: sparseCM12SharpeningTraceSteps(values.sharpeningTraceSteps),
    sharpeningStrength: sparseCM12SharpeningStrength(values.sharpeningStrength),
    // Capability, not a tuning knob: it only reserves the journal region so a
    // later frame can be armed. Off by default, so a solver that never asked
    // for the film pays nothing — not a float of state, not a dispatch.
    pressureJournal: values.pressureJournal === true || values.pressureJournal === "on",
  };
}

export const adaptiveMassMethod: SimulationMethod = {
  composition: resolveMethodComposition(),
  resolveComposition: values => resolveMethodComposition(values),
  id: "adaptive-volume",
  label: "Sparse Geometric",
  shortLabel: "Sparse Geometric",
  badge: "GEOMETRIC",
  description: "Sparse adaptive fluid with a shared level-set surface and conservative volume transport.",
  detail: "Sparse Geometric stores the liquid surface at adaptive resolution and transports liquid volume conservatively over each frame. Pressure, surface detail and rendering share the accepted level set. Volume sharpening redistributes liquid toward that surface, and pressure releases temporary excess. The sparse grid refines and coarsens with the flow while supporting live scene edits and rigid bodies.",
  backend: "webgpu",
  resource: {
    id: "fluid.adaptive-volume",
    lane: "fluid",
    label: "Sparse Geometric fluid authority",
    provides: ["fluid-authority", "water-presentation"],
    blocks: "transport",
    phaseCopy: {
      planning: "Planning the scene's sparse 4³/8³ tile atlas.",
      "adaptive-topology": "Building resident tiles, neighbours, and conservative seam ports.",
      allocation: "Allocating compact fluid authority and water presentation resources.",
      warmup: "Uploading and fencing the initial Sparse Geometric state.",
      attach: "Attaching the warmed adaptive solver atomically.",
    },
  },
  qualityLabels: {
    balanced: "Sparse graded 1³→16³",
    high: "Sparse graded 1³→16³",
    ultra: "Sparse graded 1³→16³",
  },
  showQualityControl: false,
  // Body scenes reserve sparse solid-fraction fields alongside their fluid
  // authority. A bodyless paper-scale scene omits that substantial arena, so
  // crossing between an empty and non-empty roster rebuilds once.
  capabilities: { volumeRendering: true, sparseWorld: true },
  pressureJournal: pressureCaptureDescriptor(sparseCM12PressureIterations, SPARSE_CM12_PRESSURE_JOURNAL_SNAPSHOTS),
  stageLenses: SPARSE_CM12_LENSES,
  // The ten coherence ("dirty") views are deliberately absent: they were more
  // than half this method's picker and every one of them answers a question
  // about the scheduler rather than about the water. Their overlay modes still
  // resolve — see `sparse-cm12-dirty-visualizations.ts`.
  supportedFieldModes: ["structure", "resolution", "density", "volume-levelset", "cfl", "speed", "phi", "pressure",
    "tracers", "face-velocity",
    // Listed unconditionally rather than gated on the reservation: a view that
    // vanished from the picker would be indistinguishable from one that does
    // not exist, and the reason it is empty — the film was never reserved — is
    // exactly what the reader needs told.
    "pressure-journal-residual", "pressure-journal-pressure",
    "pressure-journal-preconditioned", "pressure-journal-direction"],
  params,
  runtimeParamKeys: ADAPTIVE_MASS_RUNTIME_PARAM_KEYS,
  pipelineGraph: async () => ADAPTIVE_MASS_FLUID_PIPELINE,
  pressureMapping: "Every live Sparse Geometric step solves one globally coupled composite pressure system over regular faces and conservative 2:1 seam ports using one-reduction sparse MGPCG.",
  normalizeValues: (values) => {
    const { activitySignals: _activitySignals, ...normalizedActivity } = activityPolicy(values);
    const executionBackend = resolvePhysicsExecutionBackend(values);
    const parsedFineResolution = executionBackend === "cpu"
      ? 8 : brickFineResolution(values.brickFineResolution);
    const fineResolution: SparseBrickFineResolution = parsedFineResolution;
    return {
      ...values,
      physicsExecutionBackend: executionBackend,
      brickFineResolution: String(fineResolution),
      presentationPageResolution: String(fineResolution),
      maximumMacroSpanBricks:
        String(maximumMacroSpanBricks(values.maximumMacroSpanBricks) ?? "auto"),
      selectorMode: selectorMode(values.selectorMode),
      surfaceFineRings: boundedInteger(values.surfaceFineRings, 1, 1, 8),
      timeStep: values.timeStep === "scene" ? "scene" : "paper",
      gammaDiffusion: "off",
      surfaceSharpening: values.surfaceSharpening === "off" ? "off" : "on",
      presentationColumnHeight: values.presentationColumnHeight === "off" ? "off"
        : values.presentationColumnHeight === "on" ? "on" : "auto",
      pressureIterations: sparseCM12PressureIterations(values.pressureIterations),
      pressureRelativeTolerance:
        sparseCM12PressureRelativeTolerance(values.pressureRelativeTolerance),
      sharpeningDistance: sparseCM12SharpeningDistance(values.sharpeningDistance),
      sharpeningTraceSteps: sparseCM12SharpeningTraceSteps(values.sharpeningTraceSteps),
      sharpeningStrength: sparseCM12SharpeningStrength(values.sharpeningStrength),
      ...normalizedActivity,
    };
  },
  effectiveStep_s: (_scene, values) =>
    values.timeStep !== "scene" ? CM12_PAPER_DT_S : undefined,
  presetFor: () => {
    const { activitySignals: _activitySignals, ...activityDefaults } =
      SPARSE_CM12_ACTIVITY_POLICY;
    return {
      brickFineResolution: "8",
      presentationPageResolution: "8",
      maximumMacroSpanBricks: "auto",
      selectorMode: "coarse-first",
      surfaceFineRings: 1,
      timeStep: "paper",
      gammaDiffusion: "off",
      surfaceSharpening: "on",
      presentationColumnHeight: "auto",
      pressureIterations: SPARSE_CM12_PRESSURE_ITERATIONS,
      pressureRelativeTolerance: SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE,
      sharpeningDistance: SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
      sharpeningTraceSteps: SPARSE_CM12_SHARPENING_TRACE_STEPS,
      sharpeningStrength: SPARSE_CM12_SHARPENING_STRENGTH,
      ...activityDefaults,
    };
  },
  diagnosticRows: adaptiveMassDiagnosticRows,
  harness: async () => (await import("./harness")).adaptiveMassHarnessPlugin,
  createSolverAsync: (
    device,
    scene,
    quality,
    values,
    onRigidLoads,
    onProgress,
    signal,
  ) => {
    const options = adaptiveMassSolverOptions(values);
    if ((options.brickFineResolution !== 4 && options.brickFineResolution !== 8
        && options.brickFineResolution !== 16)
      || options.presentationPageResolution !== options.brickFineResolution) {
      return Promise.reject(new RangeError(
        "Sparse CM12 production requires a matched B4/P4, B8/P8, or B16/P16 profile",
      ));
    }
    if (resolvePhysicsExecutionBackend(values) === "cpu") {
      return RustFluid3DGPUSolverAdapter.create(device, scene, {
        quality,
        methodValues: values,
      });
    }
    return WebGPUAdaptiveMassSolver.createAsync(
      device,
      scene,
      quality,
      onRigidLoads,
      options,
      onProgress,
      signal,
    );
  },
};
