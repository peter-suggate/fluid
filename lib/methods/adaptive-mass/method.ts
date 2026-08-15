import type {
  MethodParamSpec,
  MethodParamValues,
  SimulationMethod,
} from "../../core/method-contract";
import { CM12_PAPER_DT_S } from "../../core/cm12-numerics";
import { adaptiveMassDiagnosticRows } from "./adaptive-mass-diagnostics";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from "./adaptive-mass-frame-pipeline";
import {
  sparseCM12ActivityPolicy,
  sparseCM12SharpeningDistance,
  sparseCM12SharpeningTraceSteps,
  SPARSE_CM12_ACTIVITY_POLICY,
  SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
  SPARSE_CM12_SHARPENING_TRACE_STEPS,
  type SparseCM12ActivityPolicy,
} from "./webgpu-sparse-cm12-resident";
import { WebGPUAdaptiveMassSolver } from "./webgpu-adaptive-mass-solver";

export type AdaptiveMassResolutionMode = "adaptive" | "all-fine" | "all-coarse";

/** Initial sparse-resolution split consumed by the interactive solver factory. */
export interface AdaptiveMassSolverOptions {
  readonly resolutionMode: AdaptiveMassResolutionMode;
  readonly fineTileResolution: 8;
  readonly coarseTileResolution: 4;
  readonly surfaceFineRings?: number;
  readonly receiverSupportRings?: number;
  readonly receiverFloor?: "auto" | 1 | 2 | 4 | 8;
  readonly activityPolicy?: SparseCM12ActivityPolicy;
  /** Omitted only by direct diagnostic constructors, which retain scene-step behavior. */
  readonly timeStep?: "paper" | "scene";
  /** CM12 Algorithm 2's D, in finest cells. Omitted constructors use the 3.1 upper paper bound. */
  readonly sharpeningDistance?: number;
  /** Forward-Euler substeps TraceAlongField may spend reaching D. */
  readonly sharpeningTraceSteps?: number;
}

const params: MethodParamSpec[] = [
  {
    kind: "select",
    key: "resolutionMode",
    label: "Resolution policy",
    default: "adaptive",
    tier: "coarse",
    options: [
      { value: "adaptive", label: "Adaptive · 1³ / 2³ / 4³ / 8³" },
      { value: "all-fine", label: "All fine · 8³" },
      { value: "all-coarse", label: "All coarse · 4³" },
    ],
    hint: "Fixed modes keep every resident and newly activated world tile at one rung. They provide matched-resolution parity lanes against fine or reduced Uniform CM12.",
  },
  {
    kind: "select",
    key: "selectorMode",
    label: "Adaptive criterion",
    default: "surface",
    tier: "coarse",
    update: "runtime",
    options: [
      { value: "surface", label: "Surface distance" },
      { value: "activity", label: "Surface + activity" },
    ],
    hint: "The default makes every brick touched by the free surface (or thin liquid) 8³ and coarsens outward under 2:1. Activity mode additionally uses velocity, deformation, temporal change and restriction detail.",
  },
  {
    kind: "select",
    key: "receiverFloor",
    label: "Created-region floor",
    default: "auto",
    tier: "coarse",
    options: [
      { value: "auto", label: "Auto · dam 4³" },
      { value: "1", label: "1³" },
      { value: "2", label: "2³" },
      { value: "4", label: "4³" },
      { value: "8", label: "8³" },
    ],
    hint: "Structural bootstrap floor for receiver capacity. The live GPU scheduler subsequently splits or merges existing receivers from current surface evidence.",
  },
  {
    kind: "number", key: "surfaceFineRings", label: "Initial fine surface band",
    default: 1, tier: "fine", unit: "bricks", min: 1, max: 8, step: 1, digits: 0,
    hint: "Structural count of occupied face-distance rings initialized at 8³ around the authored free surface; the 4³/2³/1³ skirt follows outside it.",
  },
  {
    kind: "number", key: "receiverSupportRings", label: "Receiver apron reach",
    default: 9, tier: "fine", unit: "bricks", min: 1, max: 24, step: 1, digits: 0,
    hint: "Structural radius of pre-created sparse receiver capacity. This bounds how far fluid can travel before a rebuild; it does not make empty world volume resident.",
  },
  {
    kind: "select",
    key: "timeStep",
    label: "Time step",
    default: "paper",
    tier: "coarse",
    options: [
      { value: "paper", label: "Paper · 1/30 s large steps" },
      { value: "scene", label: "Scene · authored maxDt" },
    ],
    update: "runtime",
    hint: "Sparse CM12 defaults to the same exact 1/30 s operating step as Uniform CM12. Scene mode is retained for matched-step validation and explicit timestep overrides.",
  },
  {
    kind: "number",
    key: "sharpeningDistance",
    label: "Mass-return distance",
    default: SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
    tier: "fine",
    unit: "cells",
    min: 0.1,
    max: 3.1,
    step: 0.1,
    digits: 1,
    update: "runtime",
    hint: "Algorithm 2's D: how far TraceAlongField may follow grad(rho) toward the 0.5 iso-contour before depositing. The paper uses 1.1 to 3.1 cells and says raising it visually resembles surface tension; below 1.1 the removed mass stays essentially where it was taken.",
  },
  {
    kind: "number",
    key: "sharpeningTraceSteps",
    label: "Trace substeps",
    default: SPARSE_CM12_SHARPENING_TRACE_STEPS,
    tier: "fine",
    unit: "substeps",
    min: 1,
    max: 16,
    step: 1,
    digits: 0,
    update: "runtime",
    hint: "How many forward-Euler substeps the trace may spend. Each is half a cell, so the reach is whichever of D and half the substep count is smaller — at the default 7 the substeps are not the binding constraint anywhere in the paper's D range, and lowering them deliberately shortens the trace along a curving gradient.",
  },
  {
    kind: "number", key: "finestTravelCells", label: "8³ velocity floor",
    default: SPARSE_CM12_ACTIVITY_POLICY.finestTravelCells, tier: "fine", update: "runtime",
    unit: "cells/step", min: 0.05, max: 4, step: 0.05, digits: 2,
    hint: "Any occupied brick whose maximum accepted displacement reaches this threshold targets 8³.",
  },
  {
    kind: "number", key: "fourTravelCells", label: "4³ velocity floor",
    default: SPARSE_CM12_ACTIVITY_POLICY.fourTravelCells, tier: "fine", update: "runtime",
    unit: "cells/step", min: 0, max: 2, step: 0.05, digits: 2,
    hint: "Displacement threshold for a 4³ minimum. Normalization keeps it no higher than the 8³ threshold.",
  },
  {
    kind: "number", key: "twoTravelCells", label: "2³ velocity floor",
    default: SPARSE_CM12_ACTIVITY_POLICY.twoTravelCells, tier: "fine", update: "runtime",
    unit: "cells/step", min: 0, max: 1, step: 0.025, digits: 3,
    hint: "Displacement threshold for a 2³ minimum; slower non-surface bulk may target 1³.",
  },
  {
    kind: "number", key: "frontLookaheadSteps", label: "Front lookahead",
    default: SPARSE_CM12_ACTIVITY_POLICY.frontLookaheadSteps, tier: "fine", update: "runtime",
    unit: "steps", min: 1, max: 32, step: 1, digits: 0,
    hint: "Sweeps surface characteristics this many accepted steps ahead when selecting and activating receiver support. Four matches the default topology cadence.",
  },
  {
    kind: "number", key: "thinFeatureCells", label: "Thin-feature floor",
    default: SPARSE_CM12_ACTIVITY_POLICY.thinFeatureCells, tier: "fine", update: "runtime",
    unit: "cells", min: 0.25, max: 8, step: 0.25, digits: 2,
    hint: "Liquid exposed on both sides of an axis and thinner than this represented width targets 8³.",
  },
  {
    kind: "number", key: "thinFeatureDensity", label: "Thin density cutoff",
    default: SPARSE_CM12_ACTIVITY_POLICY.thinFeatureDensity, tier: "fine", update: "runtime",
    unit: "ρ", min: 0, max: 0.25, step: 0.005, digits: 3,
    hint: "Minimum density allowed to pin a thin feature fine. Zero uses CM12's dry threshold; raise it to ignore increasingly dilute residue.",
  },
  {
    kind: "number", key: "surfaceDensityMinimum", label: "Surface density low",
    default: SPARSE_CM12_ACTIVITY_POLICY.surfaceDensityMinimum, tier: "fine", update: "runtime",
    unit: "ρ", min: 0, max: 0.49, step: 0.01, digits: 2,
    hint: "Lower density bound for partial-cell surface evidence. Composite rows crossing rho=.5 remain surface evidence independently.",
  },
  {
    kind: "number", key: "surfaceDensityMaximum", label: "Surface density high",
    default: SPARSE_CM12_ACTIVITY_POLICY.surfaceDensityMaximum, tier: "fine", update: "runtime",
    unit: "ρ", min: 0.51, max: 1, step: 0.01, digits: 2,
    hint: "Upper density bound for partial-cell surface evidence.",
  },
  {
    kind: "number", key: "detailTolerance", label: "Detail tolerance",
    default: SPARSE_CM12_ACTIVITY_POLICY.detailTolerance, tier: "fine", update: "runtime",
    unit: "ρ", min: 0.005, max: 0.5, step: 0.005, digits: 3,
    hint: "Maximum 2x2x2 restriction error allowed before fine detail vetoes demotion.",
  },
  {
    kind: "number", key: "topologyCadenceSteps", label: "Topology cadence",
    default: SPARSE_CM12_ACTIVITY_POLICY.topologyCadenceSteps, tier: "fine", update: "runtime",
    unit: "steps", min: 1, max: 32, step: 1, digits: 0,
    hint: "Accepted steps between GPU topology planning epochs. Surface refinements enter the urgent lane immediately; ordinary coarsening is prepared incrementally.",
  },
  {
    kind: "number", key: "prepareBricksPerFrame", label: "Topology work budget",
    default: SPARSE_CM12_ACTIVITY_POLICY.prepareBricksPerFrame,
    tier: "fine", update: "runtime",
    unit: "bricks/frame", min: 1, max: 256, step: 1, digits: 0,
    hint: "Maximum ordinary split/merge preparations the GPU round-robin lane starts per frame. Surface and thin-fluid promotions use a separate urgent lane so a moving front cannot wait behind coarsening.",
  },
  {
    kind: "number", key: "promoteEpochs", label: "Promotion persistence",
    default: SPARSE_CM12_ACTIVITY_POLICY.promoteEpochs, tier: "fine", update: "runtime",
    unit: "epochs", min: 1, max: 16, step: 1, digits: 0,
    hint: "Hot topology epochs required for ordinary promotion. Surface, thin-fluid, velocity-floor, and emergency requests bypass this delay.",
  },
  {
    kind: "number", key: "demoteEpochs", label: "Demotion persistence",
    default: SPARSE_CM12_ACTIVITY_POLICY.demoteEpochs, tier: "fine", update: "runtime",
    unit: "epochs", min: 1, max: 32, step: 1, digits: 0,
    hint: "Consecutive quiet topology epochs required before requesting one rung coarser.",
  },
  {
    kind: "number", key: "promoteScore", label: "Promotion score",
    default: SPARSE_CM12_ACTIVITY_POLICY.promoteScore, tier: "fine", update: "runtime",
    unit: "", min: 0, max: 1, step: 0.025, digits: 3,
    hint: "Normalized deformation/temporal/detail activity needed to count an epoch as hot.",
  },
  {
    kind: "number", key: "demoteScore", label: "Demotion score",
    default: SPARSE_CM12_ACTIVITY_POLICY.demoteScore, tier: "fine", update: "runtime",
    unit: "", min: 0, max: 1, step: 0.025, digits: 3,
    hint: "Maximum normalized activity allowed to count an epoch as quiet.",
  },
  {
    kind: "number", key: "emergencyScore", label: "Emergency score",
    default: SPARSE_CM12_ACTIVITY_POLICY.emergencyScore, tier: "fine", update: "runtime",
    unit: "", min: 0, max: 1, step: 0.025, digits: 3,
    hint: "Normalized activity that requests immediate one-rung promotion without waiting for persistence.",
  },
];

/**
 * The controls a live Sparse CM12 solver adopts.
 *
 * Clock, Sec. 3.5 trace scalars, and GPU candidate-policy values enter the
 * next advance through a small uniform. The per-frame preparation budget is a
 * GPU scheduler limit, not a host worklist: `advanceTo` still encodes the same
 * fixed/indirect dispatch sequence without reading queue state back.
 */
export const ADAPTIVE_MASS_RUNTIME_PARAM_KEYS = Object.freeze([
  "selectorMode",
  "timeStep",
  "sharpeningDistance",
  "sharpeningTraceSteps",
  "finestTravelCells", "fourTravelCells", "twoTravelCells",
  "frontLookaheadSteps", "thinFeatureCells", "thinFeatureDensity",
  "surfaceDensityMinimum", "surfaceDensityMaximum", "detailTolerance",
  "topologyCadenceSteps", "prepareBricksPerFrame", "promoteEpochs", "demoteEpochs",
  "promoteScore", "demoteScore", "emergencyScore",
] as const);

const resolutionMode = (value: unknown): AdaptiveMassResolutionMode =>
  value === "all-fine" || value === "all-coarse" ? value : "adaptive";

const boundedInteger = (value: unknown, fallback: number, minimum: number, maximum: number) =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(maximum, Math.max(minimum, Math.round(value))) : fallback;

const receiverFloor = (value: unknown): AdaptiveMassSolverOptions["receiverFloor"] =>
  value === "1" ? 1 : value === "2" ? 2 : value === "4" ? 4 : value === "8" ? 8 : "auto";

const selectorMode = (value: unknown): "surface" | "activity" =>
  value === "activity" ? "activity" : "surface";

const activityPolicy = (values: MethodParamValues): SparseCM12ActivityPolicy =>
  sparseCM12ActivityPolicy({
    ...values,
    activitySignals: selectorMode(values.selectorMode) === "activity",
  });

export function adaptiveMassSolverOptions(
  values: MethodParamValues,
): AdaptiveMassSolverOptions {
  return {
    resolutionMode: resolutionMode(values.resolutionMode),
    fineTileResolution: 8,
    coarseTileResolution: 4,
    surfaceFineRings: boundedInteger(values.surfaceFineRings, 1, 1, 8),
    receiverSupportRings: boundedInteger(values.receiverSupportRings, 9, 1, 24),
    receiverFloor: receiverFloor(values.receiverFloor),
    activityPolicy: activityPolicy(values),
    timeStep: values.timeStep === "scene" ? "scene" : "paper",
    sharpeningDistance: sparseCM12SharpeningDistance(values.sharpeningDistance),
    sharpeningTraceSteps: sparseCM12SharpeningTraceSteps(values.sharpeningTraceSteps),
  };
}

export const adaptiveMassMethod: SimulationMethod = {
  id: "adaptive-mass",
  label: "Sparse CM12",
  shortLabel: "Sparse CM12",
  badge: "SPARSE CM12",
  description: "Sparse 1³/2³/4³/8³ world-brick expansion of the uniform CM12 mass-conserving method.",
  detail: "Sparse CM12 maps any authored scene into a fixed-world-space GPU brick atlas and couples graded neighbours through shared conservative transport and a global composite pressure solve. Runtime surface-distance requests for existing bricks are measured, 2:1-closed and conservatively transferred into GPU-authored physical generations; urgent surface refinement bypasses the budgeted round-robin coarsening lane. Transport, pressure, projection and presentation all consume the accepted worklists.",
  backend: "webgpu",
  resource: {
    id: "fluid.sparse-cm12",
    lane: "fluid",
    label: "Sparse CM12 fluid authority",
    provides: ["fluid-authority", "water-presentation"],
    blocks: "transport",
    phaseCopy: {
      planning: "Planning the scene's sparse 4³/8³ tile atlas.",
      "adaptive-topology": "Building resident tiles, neighbours, and conservative seam ports.",
      allocation: "Allocating compact fluid authority and water presentation resources.",
      warmup: "Uploading and fencing the initial Sparse CM12 state.",
      attach: "Attaching the warmed adaptive solver atomically.",
    },
  },
  qualityLabels: {
    balanced: "Sparse graded 1³→8³",
    high: "Sparse graded 1³→8³",
    ultra: "Sparse graded 1³→8³",
  },
  showQualityControl: false,
  // `adoptsRigidRosterShape` for the opposite reason to Uniform CM12's: this
  // method has no rigid system at all — `_onRigidLoads` is ignored, `advanceTo`
  // voids its bodies, and nothing under lib/methods/adaptive-mass so much as
  // names `scene.rigidBodies` — so the roster sizes none of its allocations.
  // Keeping the first body in the solver key restarted the scene to buy an
  // identical solver, and a body that the sparse water cannot see is no less
  // invisible after a rebuild than before one.
  capabilities: { volumeRendering: true, adoptsRigidRosterShape: true },
  supportedFieldModes: ["structure", "resolution", "density", "cfl", "speed", "phi", "pressure"],
  params,
  runtimeParamKeys: ADAPTIVE_MASS_RUNTIME_PARAM_KEYS,
  pipelineGraph: async () => ADAPTIVE_MASS_FLUID_PIPELINE,
  pressureMapping: "Every live Sparse CM12 step solves one globally coupled composite pressure system over regular faces and conservative 2:1 seam ports using matrix-free Jacobi-PCG.",
  normalizeValues: (values) => {
    const { activitySignals: _activitySignals, ...normalizedActivity } = activityPolicy(values);
    return {
      ...values,
      resolutionMode: resolutionMode(values.resolutionMode),
      selectorMode: selectorMode(values.selectorMode),
      receiverFloor: String(receiverFloor(values.receiverFloor)),
      surfaceFineRings: boundedInteger(values.surfaceFineRings, 1, 1, 8),
      receiverSupportRings: boundedInteger(values.receiverSupportRings, 9, 1, 24),
      timeStep: values.timeStep === "scene" ? "scene" : "paper",
      sharpeningDistance: sparseCM12SharpeningDistance(values.sharpeningDistance),
      sharpeningTraceSteps: sparseCM12SharpeningTraceSteps(values.sharpeningTraceSteps),
      ...normalizedActivity,
    };
  },
  effectiveStep_s: (_scene, values) =>
    values.timeStep !== "scene" ? CM12_PAPER_DT_S : undefined,
  presetFor: () => {
    const { activitySignals: _activitySignals, ...activityDefaults } =
      SPARSE_CM12_ACTIVITY_POLICY;
    return {
      resolutionMode: "adaptive",
      selectorMode: "surface",
      receiverFloor: "auto",
      surfaceFineRings: 1,
      receiverSupportRings: 9,
      timeStep: "paper",
      sharpeningDistance: SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
      sharpeningTraceSteps: SPARSE_CM12_SHARPENING_TRACE_STEPS,
      ...activityDefaults,
    };
  },
  diagnosticRows: adaptiveMassDiagnosticRows,
  createSolverAsync: (
    device,
    scene,
    quality,
    values,
    onRigidLoads,
    onProgress,
    signal,
  ) => WebGPUAdaptiveMassSolver.createAsync(
    device,
    scene,
    quality,
    onRigidLoads,
    adaptiveMassSolverOptions(values),
    onProgress,
    signal,
  ),
};
