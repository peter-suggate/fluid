import {
  uniformDensityPostProcessingEnabled,
  WebGPUUniformReferenceSolver,
  type WebGPUUniformReferenceOptions,
} from "../webgpu-uniform-reference";
import {
  numberValue,
  type MethodParamSpec,
  type MethodParamValues,
  type SimulationMethod,
} from "./types";
import type { SceneDescription } from "../model";

export const UNIFORM_RUNTIME_PARAM_KEYS = Object.freeze([
  "gammaDiffusion",
  "gammaDiffusionIterations",
  "densitySharpening",
  "sharpeningMassCorrection",
  "sharpeningStrength",
  "sharpeningDistance",
  "solidExcessCorrection",
  "rigidCoupling",
  "velocityTransport",
  "timeStep",
  "densityPostProcessing",
] as const);

const runtimeUpdate = { update: "runtime" as const };

const params: MethodParamSpec[] = [
  {
    kind: "select",
    key: "activeRegion",
    label: "Active-region dispatch",
    default: "on",
    tier: "coarse",
    options: [
      { value: "on", label: "On · sparse GPU work box" },
      { value: "off", label: "Off · dense control" },
    ],
    hint: "Rebuilds the solver with GPU-resident active bounds or the original full-lattice dispatches. Off is the production A/B control.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "gammaDiffusion",
    label: "Gamma diffusion",
    default: "on",
    tier: "fine",
    options: [
      { value: "on", label: "On · ordered pair diffusion" },
      { value: "off", label: "Off · retain transported gamma" },
    ],
    hint: "Ablates Sec. 3.4's ordered pair equalization. The conservative density transport still publishes a complete rho/gamma state for downstream stages.",
  },
  {
    ...runtimeUpdate,
    kind: "number",
    key: "gammaDiffusionIterations",
    label: "Gamma iterations",
    default: 7,
    tier: "fine",
    unit: "iterations",
    min: 1,
    max: 7,
    step: 1,
    digits: 0,
    hint: "Each paper iteration is six even/odd axis passes. One is cheapest; seven is the robust high-compression setting.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "densitySharpening",
    label: "Interface sharpening",
    default: "on",
    tier: "fine",
    options: [
      { value: "on", label: "On · Sec. 3.5" },
      { value: "off", label: "Off · advected density" },
    ],
    hint: "Ablates the local Sec. 3.5 density correction. Turning it off bypasses both sharpening and its dependent mass-return stage.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "sharpeningMassCorrection",
    label: "Sharpening mass return",
    default: "on",
    tier: "fine",
    options: [
      { value: "on", label: "On · local conservative return" },
      { value: "off", label: "Off · raw density correction" },
    ],
    hint: "Controls Algorithm 2 separately from the density correction. Off keeps the sharpened field but omits the scatter/resolve that returns removed mass locally.",
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
    ...runtimeUpdate,
    kind: "select",
    key: "solidExcessCorrection",
    label: "Partial-solid excess",
    default: "on",
    tier: "fine",
    options: [
      { value: "on", label: "On · Sec. 3.6" },
      { value: "off", label: "Off · retain cut-cell excess" },
    ],
    hint: "Ablates the conservative redistribution of density that exceeds a cut cell's open fraction.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "rigidCoupling",
    label: "Rigid coupling",
    default: "on",
    tier: "fine",
    options: [
      { value: "on", label: "On · two-way coupling" },
      { value: "off", label: "Off · fluid-only motion" },
    ],
    hint: "Disables fluid/body momentum exchange and rigid integration while retaining the bodies as solid boundaries.",
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
  {
    ...runtimeUpdate,
    kind: "select",
    key: "velocityTransport",
    label: "Velocity advection",
    default: "semi-lagrangian",
    tier: "coarse",
    options: [
      { value: "semi-lagrangian", label: "Semi-Lagrangian · one pass" },
      { value: "maccormack", label: "Bounded MacCormack · three passes" },
    ],
    hint: "Semi-Lagrangian uses the original single backward-trace update. Bounded MacCormack adds a forward prediction, predicted-field extension, reverse trace, and local-extrema-limited correction.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "timeStep",
    label: "Time step",
    default: "paper",
    tier: "coarse",
    options: [
      { value: "paper", label: "Paper · 1/30 s large steps" },
      { value: "scene", label: "Scene · authored maxDt" },
    ],
    hint: "Chentanez-Müller run dt=1/30 s (CFL 8-25) in every example; Sec. 3.5 sharpening only balances transport diffusion at that per-step dose. Scene-step mode exists for matched-dt comparison lanes and dilutes the interface at small dt.",
  },
  {
    ...runtimeUpdate,
    kind: "select",
    key: "densityPostProcessing",
    label: "Sub-grid rendering",
    default: "off",
    tier: "fine",
    options: [
      { value: "scene", label: "Scene · Sec. 3.8 where needed" },
      { value: "off", label: "Wall films only" },
      { value: "on", label: "Wall films + Sec. 3.8" },
    ],
    hint: "Render-only: mass-proportional sheets against walls and solids are always reconstructed. Scene/On additionally enable the paper's Sec. 3.8 global reconstruction. Neither feeds simulation physics.",
  },
];

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
    activeRegion: values.activeRegion !== "off",
    densitySharpening: values.densitySharpening !== "off",
    sharpeningMassCorrection: values.sharpeningMassCorrection !== "off",
    gammaDiffusionIterations: values.gammaDiffusion === "off"
      ? 0 : whole("gammaDiffusionIterations"),
    sharpeningStrength: numberValue(values, params, "sharpeningStrength"),
    sharpeningDistance: numberValue(values, params, "sharpeningDistance"),
    solidExcessCorrection: values.solidExcessCorrection !== "off",
    rigidCoupling: values.rigidCoupling !== "off",
    pressureSchedule: {
      fullCycles: whole("pressureFullCycles"),
      vCycles: whole("pressureVCycles"),
      preSweeps: whole("pressureSweeps"),
      postSweeps: whole("pressureSweeps"),
    },
    densityPostProcessing: uniformDensityPostProcessingEnabled(
      values.densityPostProcessing,
      scene?.sceneId,
    ),
    timeStep: values.timeStep === "scene" ? "scene" : "paper",
    velocityTransport: values.velocityTransport === "maccormack"
      ? "maccormack" : "semi-lagrangian",
  };
}

export const uniformMethod: SimulationMethod = {
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
  params,
  runtimeParamKeys: UNIFORM_RUNTIME_PARAM_KEYS,
  pressureMapping: "CM11a uses 3 Full-Cycles, 4 V-Cycles, and six pre/post PRBGS sweeps (two above the paper's shallow-grid schedule for deep-hierarchy convergence).",
  presetFor: () => ({}),
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
