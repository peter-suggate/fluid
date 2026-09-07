import type { MethodParamSpec, MethodParamValues } from "../../../../core/method-contract";
import type { FluidStageControl } from "../../../../core/fluid-pipeline";
import type { FeatureDefinition } from "../../../../framework/composition";
import { SPARSE_CM12_ACTIVITY_POLICY } from "./policy";

export const ADAPTIVITY_PARAMS: MethodParamSpec[] = [
  { kind: "number", key: "energyThreshold", label: "Finest kinetic energy",
    default: SPARSE_CM12_ACTIVITY_POLICY.energyThreshold, tier: "fine", update: "runtime", unit: "m²/s²",
    min: 0.01, max: 100, step: 0.1, digits: 2,
    hint: "Specific kinetic energy ½|u|² requesting the finest rung. Lower rungs use dyadic speed thresholds.",
  },
  { kind: "number", key: "curvatureTolerance", label: "Curvature tolerance",
    default: SPARSE_CM12_ACTIVITY_POLICY.curvatureTolerance, tier: "fine", update: "runtime", unit: "κh",
    min: 0.02, max: 2, step: 0.01, digits: 2,
    hint: "Maximum surface normal variation per cell. Smaller values preserve finer curved liquid geometry. Static solid restriction floors remain active.",
  },
  { kind: "number", key: "anticipationSeconds", label: "Impact lookahead",
    default: SPARSE_CM12_ACTIVITY_POLICY.anticipationSeconds, tier: "fine", update: "runtime", unit: "s",
    min: 0, max: 2, step: 0.05, digits: 2,
    hint: "Predict approaching liquid from its accepted velocity over this horizon, refining receivers before contact.",
  },
  { kind: "number", key: "anticipationRadiusBricks", label: "Impact search radius",
    default: SPARSE_CM12_ACTIVITY_POLICY.anticipationRadiusBricks, tier: "fine", update: "runtime", unit: "bricks",
    min: 1, max: 6, step: 1, digits: 0,
    hint: "Bounded spatial search around a surface receiver. Increase for fast objects or longer prediction horizons; cost grows with radius cubed.",
  },
  { kind: "number", key: "surfaceQuietEpochs", label: "Surface proof persistence",
    default: SPARSE_CM12_ACTIVITY_POLICY.surfaceQuietEpochs, tier: "fine", update: "runtime", unit: "epochs",
    min: 1, max: 32, step: 1, digits: 0,
    hint: "Consecutive valid surface proofs before a coarse-first merge. Refinement is immediate.",
  },
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
    hint: "Caps macro coverage during initialization and live split/merge. Auto permits progressively larger aligned coverage as quiet siblings merge.",
  },
  {
    kind: "select",
    key: "selectorMode",
    label: "Adaptive criterion",
    default: "coarse-first",
    tier: "coarse",
    update: "runtime",
    options: [
      { value: "surface", label: "Surface distance" },
      { value: "activity", label: "Causal activity + surface proof" },
      { value: "coarse-first", label: "Coarse first" },
    ],
    hint: "Surface distance keeps interface/thin bricks at the ladder maximum. Causal activity promotes moving or unresolved liquid and lets accepted presentation output prove a one-rung surface merge. Coarse-first starts planar surfaces at B1 and refines for energy, curvature and approaching liquid.",
  },
  {
    kind: "number", key: "surfaceFineRings", label: "Initial fine surface band",
    default: 1, tier: "fine", unit: "bricks", min: 1, max: 8, step: 1, digits: 0,
    hint: "Structural count of occupied face-distance rings initialized at the ladder maximum around the authored free surface; the coarser dyadic skirt follows outside it.",
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
    hint: "Surface/front displacement threshold for a 4³ destination minimum. Normalization keeps it no higher than the 8³ threshold.",
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
    hint: "Sweeps surface characteristics this many accepted steps ahead when selecting and creating world pages. Four matches the default topology cadence.",
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
    kind: "number", key: "surfaceDisplacementToleranceCells",
    label: "Surface displacement tolerance",
    default: SPARSE_CM12_ACTIVITY_POLICY.surfaceDisplacementToleranceCells,
    tier: "fine", update: "runtime", unit: "fine cells",
    min: 0, max: 8, step: 0.05, digits: 2,
    hint: "Maximum rho=.5 edge-crossing movement accepted by each dyadic presentation proof.",
  },
  {
    kind: "number", key: "surfaceNormalToleranceDegrees",
    label: "Surface normal tolerance",
    default: SPARSE_CM12_ACTIVITY_POLICY.surfaceNormalToleranceDegrees,
    tier: "fine", update: "runtime", unit: "°",
    min: 0, max: 90, step: 1, digits: 0,
    hint: "Maximum narrow-band normal-angle error accepted by each dyadic presentation proof.",
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

export const ADAPTIVITY_MODES = (ADAPTIVITY_PARAMS.find(p => p.key === "selectorMode") as Extract<MethodParamSpec, {kind: "select"}>).options;
export const adaptivityPrimaryControls = (values: MethodParamValues) => values.selectorMode === "coarse-first"
  ? [{key: "energyThreshold", tag: "E"}, {key: "curvatureTolerance", tag: "κ"}, {key: "anticipationSeconds", tag: "T"}]
  : values.selectorMode === "activity"
    ? [{key: "finestTravelCells", tag: "U"}, {key: "surfaceDisplacementToleranceCells", tag: "Δ"}] : [];

export const adaptiveMassAdaptivityFeature: FeatureDefinition = {
  id: "simulation.adaptive-mass.adaptivity", label: "Fluid adaptivity",
  requires: ["simulation.sparse-atlas"], provides: ["simulation.resolution-policy"],
  controls: [{id:"adaptivity",label:"Fluid adaptivity",kind:"readout"}, ...ADAPTIVITY_PARAMS.map(p => ({
    id: p.key, setting: p.key, label: p.label, hint: p.hint,
    kind: p.kind === "select" ? "choice" as const : "number" as const,
    update: p.update === "runtime" ? "live" as const : "rebuild" as const,
    ...(p.kind === "number" ? {min:p.min,max:p.max,step:p.step,unit:p.unit} : {options:p.options}),
  }))],
  placements: [{slot:"scene.adaptivity",control:"adaptivity",presentation:"compact"}, ...ADAPTIVITY_PARAMS.map((p, order) => ({slot: "sim.adaptivity", control:p.key, order, presentation:"expanded" as const}))],
  variants: ADAPTIVITY_MODES.map(option => ({id:option.value, point:"simulation.adaptive-mass.adaptivity", update:"live" as const, default:option.value === "coarse-first"})),
};


/** Availability belongs to the policy, regardless of which UI host presents it. */
export function adaptivityControlEnabled(key: string, values: MethodParamValues): boolean {
  if (["energyThreshold", "curvatureTolerance", "anticipationSeconds", "anticipationRadiusBricks", "surfaceQuietEpochs"].includes(key)) return values.selectorMode === "coarse-first";
  if (key === "surfaceFineRings") return values.selectorMode !== "coarse-first";
  if (["finestTravelCells", "fourTravelCells", "twoTravelCells", "detailTolerance", "promoteEpochs", "demoteEpochs", "promoteScore", "demoteScore", "emergencyScore"].includes(key)) return values.selectorMode === "activity";
  if (["surfaceDisplacementToleranceCells", "surfaceNormalToleranceDegrees", "topologyCadenceSteps"].includes(key)) return values.selectorMode === "activity" || values.selectorMode === "coarse-first";
  return true;
}

/** Stage hosts consume the same authoritative limits, labels and choices. */
export function adaptivityStageControl(key: string): FluidStageControl {
  const spec = ADAPTIVITY_PARAMS.find(p => p.key === key);
  if (!spec) throw new Error(`Unknown adaptivity control: ${key}`);
  if (spec.kind === "select") return {kind:"param-choice",param:key,label:spec.label,hint:spec.hint,options:spec.options};
  return {kind:"param-range",param:key,label:spec.label,hint:spec.hint,unit:spec.unit ? ` ${spec.unit}` : "",min:spec.min ?? 0,max:spec.max ?? 1,step:spec.step ?? 0.01,digits:spec.digits,enabled: context => adaptivityControlEnabled(key, context.values)};
}
