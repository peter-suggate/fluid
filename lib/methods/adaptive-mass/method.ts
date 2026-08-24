import type {
  MethodParamSpec,
  MethodParamValues,
  SimulationMethod,
} from "../../core/method-contract";
import { SPARSE_CM12_LENSES } from "./sparse-cm12-stage-lenses";
import { CM12_PAPER_DT_S } from "../../core/cm12-numerics";
import { pressureJournalSchedule } from "../../core/pressure-journal";
import { SPARSE_CM12_DIRTY_OVERLAY_MODES } from "../../core/sparse-cm12-dirty-visualizations";
import { adaptiveMassDiagnosticRows } from "./adaptive-mass-diagnostics";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from "./adaptive-mass-frame-pipeline";
import {
  sparseCM12ActivityPolicy,
  sparseCM12PressureIterations,
  sparseCM12PressureRelativeTolerance,
  sparseCM12SharpeningDistance,
  sparseCM12SharpeningTraceSteps,
  SPARSE_CM12_ACTIVITY_POLICY,
  SPARSE_CM12_PRESSURE_ITERATIONS,
  SPARSE_CM12_PRESSURE_JOURNAL_SNAPSHOTS,
  SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE,
  SPARSE_CM12_SHARPENING_DISTANCE_CELLS,
  SPARSE_CM12_SHARPENING_TRACE_STEPS,
  type SparseCM12ActivityPolicy,
} from "./webgpu-sparse-cm12-resident";
import { WebGPUAdaptiveMassSolver } from "./webgpu-adaptive-mass-solver";
import type {
  SparseBrickFineResolution,
  SparseBrickResolution,
} from "./sparse-brick-atlas";

export type AdaptiveMassResolutionMode = "adaptive" | "all-fine" | "all-coarse";

/** Initial sparse-resolution split consumed by the interactive solver factory. */
export interface AdaptiveMassSolverOptions {
  readonly resolutionMode: AdaptiveMassResolutionMode;
  /** Construction-time complete dyadic ladder maximum. Defaults to 8. */
  readonly brickFineResolution?: SparseBrickFineResolution;
  /** Renderer-facing samples per presentation-page edge. Defaults to the brick maximum. */
  readonly presentationPageResolution?: SparseBrickFineResolution;
  /** Optional positive-power-of-two cap on hierarchical macro-leaf span. */
  readonly maximumMacroSpanBricks?: number;
  readonly surfaceFineRings?: number;
  readonly receiverSupportRings?: number;
  readonly receiverFloor?: "auto" | SparseBrickResolution;
  readonly activityPolicy?: SparseCM12ActivityPolicy;
  /** Omitted only by direct diagnostic constructors, which retain scene-step behavior. */
  readonly timeStep?: "paper" | "scene";
  /** CM12 Algorithm 2's D, in finest cells. Omitted constructors use the 3.1 upper paper bound. */
  readonly sharpeningDistance?: number;
  /** Forward-Euler substeps TraceAlongField may spend reaching D. */
  readonly sharpeningTraceSteps?: number;
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
  {
    kind: "select",
    key: "brickFineResolution",
    label: "Brick ladder",
    default: "8",
    tier: "coarse",
    options: [
      { value: "4", label: "1³ / 2³ / 4³ · experimental" },
      { value: "8", label: "1³ / 2³ / 4³ / 8³ · experimental" },
      { value: "16", label: "1³ / 2³ / 4³ / 8³ / 16³" },
    ],
    hint: "Selects both the adaptive ladder maximum and the matching presentation-page resolution. B8 is the production default; B4 remains experimental and B16 remains available.",
  },
  {
    kind: "select",
    key: "resolutionMode",
    label: "Resolution policy",
    default: "adaptive",
    tier: "coarse",
    options: [
      { value: "adaptive", label: "Adaptive · complete dyadic ladder" },
      { value: "all-fine", label: "All fine · ladder maximum" },
      { value: "all-coarse", label: "All coarse · one rung below maximum" },
    ],
    hint: "Fixed modes keep every resident and newly activated world tile at one rung. They provide matched-resolution parity lanes against fine or reduced Uniform CM12.",
  },
  {
    kind: "select",
    key: "maximumMacroSpanBricks",
    label: "Largest macro span",
    default: "auto",
    tier: "coarse",
    options: [
      { value: "auto", label: "Auto · largest aligned cover" },
      { value: "1", label: "1 brick · macros off" },
      { value: "2", label: "2 bricks" },
      { value: "4", label: "4 bricks" },
      { value: "8", label: "8 bricks" },
      { value: "16", label: "16 bricks" },
      { value: "32", label: "32 bricks" },
      { value: "64", label: "64 bricks" },
    ],
    hint: "Caps the edge span of immutable hierarchical leaves. Auto retains the largest aligned dyadic cover supported by the scene.",
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
    hint: "Surface distance keeps interface/thin bricks at the ladder maximum and sends submerged bricks directly to 1³ before 2:1 closure, ignoring velocity and history. Activity additionally refines moving or complex liquid.",
  },
  {
    kind: "select",
    key: "receiverFloor",
    label: "Created-region floor",
    default: "auto",
    tier: "coarse",
    options: [
      { value: "auto", label: "Auto · dam at coarse rung" },
      { value: "1", label: "1³" },
      { value: "2", label: "2³" },
      { value: "4", label: "4³" },
      { value: "8", label: "8³" },
      { value: "16", label: "16³" },
    ],
    hint: "Structural bootstrap floor for receiver capacity. The live GPU scheduler subsequently splits or merges existing receivers from current surface evidence.",
  },
  {
    kind: "number", key: "surfaceFineRings", label: "Initial fine surface band",
    default: 1, tier: "fine", unit: "bricks", min: 1, max: 8, step: 1, digits: 0,
    hint: "Structural count of occupied face-distance rings initialized at the ladder maximum around the authored free surface; the coarser dyadic skirt follows outside it.",
  },
  {
    kind: "number", key: "receiverSupportRings", label: "Receiver apron reach",
    default: 9, tier: "fine", unit: "bricks", min: 1, max: 24, step: 1, digits: 0,
    hint: "Structural radius of pre-created sparse receiver capacity at nominal detail. DETAIL edits scale its brick radius and minimum pool volume so the same physical travel remains available; it does not make unrelated empty world volume resident.",
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
    hint: "Sparse CM12 defaults to the exact 1/30 s paper step. Scene mode remains available for matched-step validation and hydrostatic probes.",
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
    hint: "Maximum sparse MGPCG iterations per pressure solve. Lower budgets trade incompressibility for frame time; 64 is the validated production operating point.",
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
    kind: "select",
    key: "pressureJournal",
    label: "Pressure film capture",
    default: "off",
    tier: "fine",
    options: [
      { value: "off", label: "Off" },
      { value: "on", label: "On · reserve the film" },
    ],
    // Structural, deliberately: the reservation is a region of the state buffer
    // and it is not small, so it cannot appear and disappear under a live
    // solver. Turning it on rebuilds once, which is the honest price of a
    // capture that is otherwise free — an armed frame is the only frame that
    // encodes a snapshot dispatch, and an unarmed one costs literally nothing.
    hint: "Reserves room to film one pressure solve, so the Pressure lab views can replay its iterations. About 192 bytes per pressure cell — a few megabytes on the mini scenes and hundreds on a large one — which is why it is off unless asked for. Reserving is not capturing: the snapshots are only written on frames a Pressure lab view is open, and the reservation alone changes no dispatch.",
  },
  {
    kind: "number", key: "finestTravelCells", label: "8³ surface travel",
    default: SPARSE_CM12_ACTIVITY_POLICY.finestTravelCells, tier: "fine", update: "runtime",
    unit: "cells/step", min: 0.05, max: 4, step: 0.05, digits: 2,
    hint: "Surface/front displacement threshold for 8³ lookahead. Fully flooded bulk does not refine merely because it translates uniformly.",
  },
  {
    kind: "number", key: "fourTravelCells", label: "4³ surface travel",
    default: SPARSE_CM12_ACTIVITY_POLICY.fourTravelCells, tier: "fine", update: "runtime",
    unit: "cells/step", min: 0, max: 2, step: 0.05, digits: 2,
    hint: "Surface/front displacement threshold for a 4³ receiver minimum. Normalization keeps it no higher than the 8³ threshold.",
  },
  {
    kind: "number", key: "twoTravelCells", label: "2³ surface travel",
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
    kind: "number", key: "residencyDensity", label: "Region density cutoff",
    default: SPARSE_CM12_ACTIVITY_POLICY.residencyDensity, tier: "fine", update: "runtime",
    unit: "ρ", min: 0.000_01, max: 0.05, step: 0.001, digits: 3,
    hint: "Minimum cell density that keeps a sparse region populated. Lower numerical residue is retired once it leaves interface support.",
  },
  {
    kind: "number", key: "residencyMassFineCells", label: "Region mass cutoff",
    default: SPARSE_CM12_ACTIVITY_POLICY.residencyMassFineCells,
    tier: "fine", update: "runtime", unit: "cells", min: 0, max: 8,
    step: 0.25, digits: 2,
    hint: "Minimum integrated liquid mass needed to keep an 8³ region populated. The default rejects fragments smaller than one full finest cell.",
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
  "pressureIterations",
  "pressureRelativeTolerance",
  "sharpeningDistance",
  "sharpeningTraceSteps",
  "finestTravelCells", "fourTravelCells", "twoTravelCells",
  "frontLookaheadSteps", "thinFeatureCells", "thinFeatureDensity", "residencyDensity",
  "residencyMassFineCells",
  "surfaceDensityMinimum", "surfaceDensityMaximum", "detailTolerance",
  "topologyCadenceSteps", "prepareBricksPerFrame", "promoteEpochs", "demoteEpochs",
  "promoteScore", "demoteScore", "emergencyScore",
] as const);

const resolutionMode = (value: unknown): AdaptiveMassResolutionMode =>
  value === "all-fine" || value === "all-coarse" ? value : "adaptive";

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

const receiverFloor = (
  value: unknown,
  maximum: SparseBrickFineResolution = 8,
): AdaptiveMassSolverOptions["receiverFloor"] => {
  const parsed = value === "1" ? 1 : value === "2" ? 2 : value === "4" ? 4
    : value === "8" ? 8 : value === "16" ? 16 : "auto";
  return parsed === "auto" ? parsed : Math.min(parsed, maximum) as SparseBrickResolution;
};

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
  const fineResolution = brickFineResolution(values.brickFineResolution);
  return {
    resolutionMode: resolutionMode(values.resolutionMode),
    brickFineResolution: fineResolution,
    presentationPageResolution:
      presentationPageResolution(values.presentationPageResolution, fineResolution),
    maximumMacroSpanBricks: maximumMacroSpanBricks(values.maximumMacroSpanBricks),
    surfaceFineRings: boundedInteger(values.surfaceFineRings, 1, 1, 8),
    receiverSupportRings: boundedInteger(values.receiverSupportRings, 9, 1, 24),
    receiverFloor: receiverFloor(values.receiverFloor, fineResolution),
    activityPolicy: activityPolicy(values),
    timeStep: values.timeStep === "scene" ? "scene" : "paper",
    pressureIterations: sparseCM12PressureIterations(values.pressureIterations),
    pressureRelativeTolerance:
      sparseCM12PressureRelativeTolerance(values.pressureRelativeTolerance),
    sharpeningDistance: sparseCM12SharpeningDistance(values.sharpeningDistance),
    sharpeningTraceSteps: sparseCM12SharpeningTraceSteps(values.sharpeningTraceSteps),
    // Capability, not a tuning knob: it only reserves the journal region so a
    // later frame can be armed. Off by default, so a solver that never asked
    // for the film pays nothing — not a float of state, not a dispatch.
    pressureJournal: values.pressureJournal === true || values.pressureJournal === "on",
  };
}

export const adaptiveMassMethod: SimulationMethod = {
  id: "adaptive-mass",
  label: "Sparse CM12",
  shortLabel: "Sparse CM12",
  badge: "SPARSE CM12",
  description: "Sparse complete-dyadic world-brick expansion of the uniform CM12 mass-conserving method.",
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
    balanced: "Sparse graded 1³→16³",
    high: "Sparse graded 1³→16³",
    ultra: "Sparse graded 1³→16³",
  },
  showQualityControl: false,
  // Body scenes reserve sparse solid-fraction fields alongside their fluid
  // authority. A bodyless paper-scale scene omits that substantial arena, so
  // crossing between an empty and non-empty roster rebuilds once.
  capabilities: { volumeRendering: true, sparseWorld: true },
  pressureJournal: {
    isReserved: (values) => values.pressureJournal === "on",
    schedule: (values) => values.pressureJournal === "on"
      ? pressureJournalSchedule(
        sparseCM12PressureIterations(values.pressureIterations),
        SPARSE_CM12_PRESSURE_JOURNAL_SNAPSHOTS,
      )
      : [],
    reserve: { parameter: "pressureJournal", value: "on" },
  },
  stageLenses: SPARSE_CM12_LENSES,
  supportedFieldModes: ["structure", "resolution", "density", "cfl", "speed", "phi", "pressure",
    "tracers", "face-velocity",
    ...SPARSE_CM12_DIRTY_OVERLAY_MODES,
    // Listed unconditionally rather than gated on the reservation: a view that
    // vanished from the picker would be indistinguishable from one that does
    // not exist, and the reason it is empty — the film was never reserved — is
    // exactly what the reader needs told.
    "pressure-journal-residual", "pressure-journal-pressure",
    "pressure-journal-preconditioned", "pressure-journal-direction"],
  params,
  runtimeParamKeys: ADAPTIVE_MASS_RUNTIME_PARAM_KEYS,
  pipelineGraph: async () => ADAPTIVE_MASS_FLUID_PIPELINE,
  pressureMapping: "Every live Sparse CM12 step solves one globally coupled composite pressure system over regular faces and conservative 2:1 seam ports using one-reduction sparse MGPCG.",
  normalizeValues: (values) => {
    const { activitySignals: _activitySignals, ...normalizedActivity } = activityPolicy(values);
    const parsedFineResolution = brickFineResolution(values.brickFineResolution);
    const fineResolution: SparseBrickFineResolution = parsedFineResolution;
    return {
      ...values,
      brickFineResolution: String(fineResolution),
      presentationPageResolution: String(fineResolution),
      maximumMacroSpanBricks:
        String(maximumMacroSpanBricks(values.maximumMacroSpanBricks) ?? "auto"),
      resolutionMode: resolutionMode(values.resolutionMode),
      selectorMode: selectorMode(values.selectorMode),
      receiverFloor: String(receiverFloor(
        values.receiverFloor, fineResolution,
      )),
      surfaceFineRings: boundedInteger(values.surfaceFineRings, 1, 1, 8),
      receiverSupportRings: boundedInteger(values.receiverSupportRings, 9, 1, 24),
      timeStep: values.timeStep === "scene" ? "scene" : "paper",
      pressureIterations: sparseCM12PressureIterations(values.pressureIterations),
      pressureRelativeTolerance:
        sparseCM12PressureRelativeTolerance(values.pressureRelativeTolerance),
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
      brickFineResolution: "8",
      presentationPageResolution: "8",
      maximumMacroSpanBricks: "auto",
      selectorMode: "surface",
      receiverFloor: "auto",
      surfaceFineRings: 1,
      receiverSupportRings: 9,
      timeStep: "paper",
      pressureIterations: SPARSE_CM12_PRESSURE_ITERATIONS,
      pressureRelativeTolerance: SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE,
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
  ) => {
    const options = adaptiveMassSolverOptions(values);
    if ((options.brickFineResolution !== 4 && options.brickFineResolution !== 8
        && options.brickFineResolution !== 16)
      || options.presentationPageResolution !== options.brickFineResolution) {
      return Promise.reject(new RangeError(
        "Sparse CM12 production requires a matched B4/P4, B8/P8, or B16/P16 profile",
      ));
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
