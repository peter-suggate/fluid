import { ALGORITHM_PARAMS } from "./features/algorithms/definition";
import { resolveMethodComposition } from "./composition";
import {
  uniformDensityPostProcessingEnabled,
  UNIFORM_FLUID_PIPELINE,
  WebGPUUniformReferenceSolver,
  type WebGPUUniformReferenceOptions,
} from "./webgpu-uniform-reference";
import { UNIFORM_PAPER_DT_S } from "./uniform-paper";
import { uniformDiagnosticRows } from "./uniform-diagnostics";
import {
  numberValue,
  type MethodParamSpec,
  type MethodParamValues,
  type SimulationMethod,
} from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";



const runtimeUpdate = { update: "runtime" as const };

const params: MethodParamSpec[] = [
  ...ALGORITHM_PARAMS,
  {
    ...runtimeUpdate, kind: "number", key: "pressureResidualTolerance",
    label: "Pressure residual tolerance", default: 10, tier: "fine",
    min: 0, max: 100, step: 0.0001, digits: 4, unit: "s⁻¹",
    hint: "Stop after a complete Full-Cycle or V-Cycle when the projected residual infinity norm is at or below this tolerance. Zero runs every configured cycle.",
  },
  {
    ...runtimeUpdate, kind: "select", key: "pressureCycleBudget",
    label: "Pressure cycle budget", default: "lagged", tier: "fine",
    options: [{ value: "lagged", label: "Lagged" }, { value: "fixed", label: "Fixed" }],
    hint: "Lagged encodes only as many multigrid cycles as the latest diagnostics sample says the solve needed, so the unused tail costs neither its GPU launch floor nor its CPU encode. Fixed always encodes the configured schedule and lets the GPU residual gate skip the remainder, which is the pre-P1 command stream.",
  },
  {
    ...runtimeUpdate, kind: "number", key: "pressureBudgetHeadroom",
    label: "Budget headroom", default: 1, tier: "fine",
    min: 0, max: 4, step: 1, digits: 0, unit: "cycles",
    hint: "Cycles the lagged budget encodes above the last observed demand. A step that used every encoded cycle without meeting tolerance doubles its budget instead, so an impact frame recovers the full schedule in one or two steps.",
  },
  {
    ...runtimeUpdate, kind: "number", key: "extensionFrontSweeps",
    label: "Extension front sweeps", default: 16, tier: "fine",
    min: 1, max: 16, step: 1, digits: 0, unit: "sweeps",
    hint: "Sec. 3.3 FIM sweep budget for the two-cell accurate band. Each sweep is an update and a dispatch-gate pass whether or not work remains; sixteen covers the band's full dependency diameter. Fewer sweeps resolve an unconverged front and leave unreached band faces to the hierarchy fill.",
  },
  {
    ...runtimeUpdate,
    kind: "number",
    key: "gammaDiffusionIterations",
    label: "Gamma iterations",
    default: 1,
    tier: "fine",
    unit: "iterations",
    min: 1,
    max: 7,
    step: 1,
    digits: 0,
    hint: "Each paper iteration is three snapshot axis passes. One is the reference default; additional repetitions deliberately apply more diffusion.",
  },
  {
    ...runtimeUpdate,
    kind: "number",
    key: "sharpeningStrength",
    label: "Sharpening strength",
    default: 1,
    tier: "fine",
    unit: "×",
    min: 0.25,
    max: 2,
    step: 0.05,
    digits: 2,
    hint: "Scales the paper's 3dt pseudo-time dose used by the local density correction.",
  },
  {
    ...runtimeUpdate,
    kind: "number",
    key: "sharpeningDistance",
    label: "Mass-return distance",
    default: 2.1,
    tier: "fine",
    unit: "cells",
    min: 0.1,
    max: 3.1,
    step: 0.1,
    digits: 1,
    hint: "Maximum Algorithm 2 gradient-trace distance D. The paper uses 1.1 to 3.1 cells; below 1.1 mass stays where it was removed, which reads as weaker surface tension.",
  },
  {
    kind: "number",
    key: "pressureFullCycles",
    label: "Pressure Full-Cycles",
    default: 3,
    tier: "fine",
    unit: "cycles",
    min: 0,
    max: 5,
    step: 1,
    digits: 0,
    hint: "CM11a Full-Cycles seed corrections from the coarsest grid upward. The paper schedule uses three. This prebuilt dispatch schedule resets the solver when changed.",
  },
  {
    kind: "number",
    key: "pressureVCycles",
    label: "Pressure V-Cycles",
    default: 4,
    tier: "fine",
    unit: "cycles",
    min: 0,
    max: 8,
    step: 1,
    digits: 0,
    hint: "Refinement V-Cycles after the Full-Cycles. The paper schedule uses four. This prebuilt dispatch schedule resets the solver when changed.",
  },
  {
    kind: "number",
    key: "pressureSweeps",
    label: "Pressure pre/post sweeps",
    default: 6,
    tier: "fine",
    unit: "sweeps",
    min: 1,
    max: 8,
    step: 1,
    digits: 0,
    hint: "Projected red-black Gauss-Seidel sweeps on each side of a multigrid coarse correction. The paper used four; six keeps deeper 64×32×64 hierarchies converged. This prebuilt dispatch schedule resets the solver when changed.",
  },
];

export const UNIFORM_RUNTIME_PARAM_KEYS = Object.freeze(params.filter(param => param.update === "runtime").map(param => param.key));

/**
 * Fixed numerical contract for the dense comparison lane.
 *
 * Keeping transport and conditioning out of the method controls makes this a
 * stable reference rather than a second experimental solver family. Its grid
 * is the scene-authored finest lattice used as the base resolution by both
 * adaptive backends.
 */
export function uniformReferenceSolverOptions(
  values: MethodParamValues,
  scene?: Pick<SceneDescription, "sceneId">,
): WebGPUUniformReferenceOptions {
  const whole = (key: string) => Math.round(numberValue(values, params, key));
  return {
    activeRegion: values.activeRegion === "on",
    densitySharpening: values.densitySharpening !== "off",
    sharpeningMassCorrection: values.sharpeningMassCorrection !== "off",
    gammaDiffusionIterations: values.gammaDiffusion === "off"
      ? 0 : whole("gammaDiffusionIterations"),
    sharpeningStrength: numberValue(values, params, "sharpeningStrength"),
    sharpeningDistance: numberValue(values, params, "sharpeningDistance"),
    solidExcessCorrection: values.solidExcessCorrection !== "off",
    rigidCoupling: values.rigidCoupling !== "off",
    extensionFrontSweeps: whole("extensionFrontSweeps"),
    pressureSchedule: {
      residualTolerance: numberValue(values, params, "pressureResidualTolerance"),
      fullCycles: whole("pressureFullCycles"),
      vCycles: whole("pressureVCycles"),
      preSweeps: whole("pressureSweeps"),
      postSweeps: whole("pressureSweeps"),
    },
    pressureCycleBudget: values.pressureCycleBudget === "fixed" ? "fixed" : "lagged",
    pressureBudgetHeadroom: whole("pressureBudgetHeadroom"),
    densityPostProcessing: uniformDensityPostProcessingEnabled(
      values.densityPostProcessing,
      scene?.sceneId,
    ),
    timeStep: values.timeStep === "scene" ? "scene" : "paper",
    velocityTransport: values.velocityTransport === "maccormack"
      ? "maccormack" : "semi-lagrangian",
    liquidOnlyVelocityAdvection: values.liquidOnlyVelocityAdvection === "on",
  };
}

export const uniformMethod: SimulationMethod = {
  composition: resolveMethodComposition(),
  resolveComposition: values => resolveMethodComposition(values),
  id: "uniform",
  label: "Uniform GPU reference",
  shortLabel: "Uniform",
  badge: "UNIFORM GPU",
  description: "Dense matched-lattice WebGPU baseline with no adaptive topology.",
  detail: "A full-depth implementation of Chentanez-Müller's mass-conserving surface-density method with persistent gamma transport, local sharpening, hierarchical velocity extension, and the CM11a separating-boundary LCP multigrid projection. It is the dense reference for Losasso and Power comparisons.",
  backend: "webgpu",
  resource: {
    id: "fluid.uniform-reference",
    lane: "fluid",
    label: "Uniform GPU fluid reference",
    provides: ["fluid-authority", "water-presentation"],
    blocks: "transport",
    phaseCopy: {
      planning: "Resolving the dense reference capabilities.",
      allocation: "Allocating matched-lattice textures and buffers.",
      "solver-pipelines": "Compiling uniform transport and pressure programs.",
      warmup: "Uploading and fencing the reference t=0 state.",
      attach: "Attaching the warmed uniform reference atomically.",
    },
  },
  qualityLabels: {
    balanced: "CM11a fixed cycles",
    high: "CM11a fixed cycles",
    ultra: "CM11a fixed cycles",
  },
  // The dense reference publishes occupancy, velocity, and a density-derived
  // surface, so only the generic dense-grid views can draw honest data; the
  // octree technique overlays would read a compact source it never produces.
  // `density` is this method's own state variable rather than a derived view,
  // which is why it is offered here and withheld from the level-set octree.
  supportedFieldModes: ["structure", "density", "cfl", "speed", "phi"],
  // Nothing here is sized from the roster: the rigid arenas are allocated at
  // GPU_RIGID_BODY_CAPACITY in the constructor, `syncBodies` re-uploads the
  // whole roster every advance, and the coupling dispatch is gated at encode
  // time on `activeBodies.length > 0`. So a body can be dropped into water
  // that is already moving and be coupled on the very next step.
  capabilities: { adoptsRigidRosterShape: true },
  params,
  runtimeParamKeys: UNIFORM_RUNTIME_PARAM_KEYS,
  pipelineGraph: async () => UNIFORM_FLUID_PIPELINE,
  // The dense lattice's own counters: cells rather than resolved rows, a
  // pressure in Pa, and the rolling work box no adaptive method has.
  diagnosticRows: uniformDiagnosticRows,
  // The paper profile is a numerical contract, not a substep ceiling: every
  // CM12 Sec. 4 example is one 1/30 s advance. `timeStep: "scene"` is the
  // opt-out that hands the clock back to the scene author.
  effectiveStep_s: (_scene, values) =>
    values.timeStep !== "scene" ? UNIFORM_PAPER_DT_S : undefined,
  pressureMapping: "CM11a uses 3 Full-Cycles, 4 V-Cycles, and six pre/post PRBGS sweeps (two above the paper's shallow-grid schedule for deep-hierarchy convergence).",
  presetFor: () => ({}),
  harness: async () => (await import("./harness")).uniformHarnessPlugin,
  createSolverAsync: (
    device,
    scene,
    quality,
    values,
    onRigidLoads,
    onProgress,
    signal,
  ) => WebGPUUniformReferenceSolver.createAsync(
    device,
    scene,
    quality,
    onRigidLoads,
    uniformReferenceSolverOptions(values, scene),
    onProgress,
    signal,
  ),
};
