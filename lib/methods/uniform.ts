import {
  uniformDensityPostProcessingEnabled,
  WebGPUUniformReferenceSolver,
  type WebGPUUniformReferenceOptions,
} from "../webgpu-uniform-reference";
import {
  type MethodParamSpec,
  type MethodParamValues,
  type SimulationMethod,
} from "./types";
import type { SceneDescription } from "../model";

const params: MethodParamSpec[] = [
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
    hint: "Chentanez-Müller run dt=1/30 s (CFL 8-25) in every example; Sec. 3.5 sharpening only balances transport diffusion at that per-step dose. Scene-step mode exists for matched-dt comparison lanes and dilutes the interface at small dt.",
  },
  {
    kind: "select",
    key: "densityPostProcessing",
    label: "Sub-grid rendering",
    default: "scene",
    tier: "fine",
    options: [
      { value: "scene", label: "Scene · Sec. 3.8 for symmetry" },
      { value: "off", label: "Off · raw paper density" },
      { value: "on", label: "On · Sec. 3.8 reconstruction" },
    ],
    hint: "Render-only: symmetric expansion uses Section 3.8 to expose sub-grid mass; other scenes retain the paper Results default (off). It never feeds simulation physics.",
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
  return {
    densitySharpening: values.densitySharpening !== "off",
    densityPostProcessing: uniformDensityPostProcessingEnabled(
      values.densityPostProcessing,
      scene?.sceneId,
    ),
    timeStep: values.timeStep === "scene" ? "scene" : "paper",
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
  supportedFieldModes: ["structure", "cfl", "speed", "phi"],
  params,
  pressureMapping: "CM11a fixes 3 Full-Cycles, 4 V-Cycles, and four pre/post PRBGS sweeps.",
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
