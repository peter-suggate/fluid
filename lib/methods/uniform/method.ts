import { UNIFORM_PARAMS as params } from "./parameters";
import { resolveMethodComposition } from "./composition";
import {
  UNIFORM_FLUID_PIPELINE,
  WebGPUUniformReferenceSolver,
} from "./webgpu-uniform-reference";
import { UNIFORM_PAPER_DT_S } from "./uniform-paper";
import { uniformDiagnosticRows } from "./uniform-diagnostics";
import {
  type MethodParamValues,
  type SimulationMethod,
} from "../../core/method-contract";
import { uniformReferenceSolverOptions } from "./uniform-options";
export { uniformReferenceSolverOptions } from "./uniform-options";



export const UNIFORM_RUNTIME_PARAM_KEYS = Object.freeze(params.filter(param => param.update === "runtime").map(param => param.key));

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
