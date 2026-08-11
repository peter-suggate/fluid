import {
  WebGPUUniformReferenceSolver,
  type WebGPUUniformReferenceOptions,
} from "../webgpu-uniform-reference";
import {
  numberValue,
  type MethodParamSpec,
  type MethodParamValues,
  type SimulationMethod,
} from "./types";

const params: MethodParamSpec[] = [
  {
    kind: "number",
    key: "pressureIterations",
    label: "Pressure sweeps",
    unit: "iterations",
    min: 16,
    max: 400,
    step: 8,
    digits: 0,
    default: 64,
    tier: "coarse",
    hint: "Weighted-Jacobi sweeps per step. This is the reference method's only algorithmic control.",
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
): WebGPUUniformReferenceOptions {
  return {
    pressureIterations: numberValue(values, params, "pressureIterations"),
  };
}

export const uniformMethod: SimulationMethod = {
  id: "uniform",
  label: "Uniform GPU reference",
  shortLabel: "Uniform",
  badge: "UNIFORM GPU",
  description: "Dense matched-lattice WebGPU baseline with no adaptive topology.",
  detail: "A full-depth uniform grid with bounded MacCormack velocity transport, conservative VOF, and weighted-Jacobi pressure projection. It exists as a transparent reference for Losasso and Power comparisons.",
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
    balanced: "64 pressure sweeps",
    high: "80 pressure sweeps",
    ultra: "96 pressure sweeps",
  },
  params,
  pressureMapping: "The quality preset selects a weighted-Jacobi sweep budget; the pressure-sweeps control overrides it directly.",
  presetFor: (quality) => ({
    pressureIterations: quality === "balanced" ? 64 : quality === "high" ? 80 : 96,
  }),
  createSolver: (device, scene, quality, values, onRigidLoads) =>
    new WebGPUUniformReferenceSolver(
      device,
      scene,
      quality,
      onRigidLoads,
      uniformReferenceSolverOptions(values),
    ),
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
    uniformReferenceSolverOptions(values),
    onProgress,
    signal,
  ),
};
