import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type SimulationMethod } from "./types";

const params: MethodParamSpec[] = [
  { kind: "number", key: "pressureIterations", label: "ICCG iterations", unit: "iterations", min: 16, max: 1024, step: 16, digits: 0, default: 96, tier: "coarse", hint: "Maximum modified-ICCG iterations; the paper stops at relative residual 1e-4. Large systems raise this floor automatically." },
  { kind: "number", key: "surfaceColumns", label: "Finest columns", unit: "columns", min: 1_000, max: 20_000, step: 500, digits: 0, default: 2_500, tier: "fine", hint: "Finest x/z lattice used by quadtree leaves and the cubic advection field." },
  { kind: "number", key: "adaptivityStrength", label: "Adaptivity", unit: "alpha", min: 0, max: 1, step: 0.05, digits: 2, default: 1, tier: "fine", hint: "Ando–Batty Eq. 38: 0 is the ordinary-grid limit; 1 permits full quadtree coarsening." }
];

export const quadtreeTallCellMethod: SimulationMethod = {
  id: "quadtree-tall-cell",
  label: "Quadtree tall cells",
  shortLabel: "Adaptive",
  badge: "QUADTREE TALL CELLS",
  description: "Narita et al. 2025: horizontally adaptive quadtree leaves with variational tall-cell pressure projection.",
  detail: "coarse-to-fine quadtree sizing from surface curvature and velocity variation, strict 2:1 adaptivity smoothing, multiple vertical tall runs, horizontally centered pressure samples, T-junction face-overlap gradients, corrected inner ghost volumes, SPD free-surface scaling, and modified ICCG(0)",
  backend: "webgpu",
  qualityLabels: { balanced: "2.5k finest columns", high: "7k finest columns", ultra: "12.5k finest columns" },
  params,
  pressureMapping: "The iteration budget is the PCG maximum; convergence is measured against the scene's relative pressure tolerance (the paper uses 1e-4).",
  presetFor: (quality) => ({ pressureIterations: quality === "balanced" ? 96 : quality === "high" ? 160 : 240, surfaceColumns: quality === "balanced" ? 2_500 : quality === "high" ? 7_000 : 12_500, adaptivityStrength: 1 }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUUniformEulerianSolver(device, scene, quality, onRigidLoads, {
    // The quadtree method follows Narita/Ando: liquid tracking is the
    // advected + redistanced level set, not the mass-conserving paper's
    // density stage, so the shared cubic advection runs without sharpening.
    densitySharpening: false,
    pressureIterations: numberValue(values, params, "pressureIterations"),
    tallCellSettings: { surfaceColumns: numberValue(values, params, "surfaceColumns") },
    quadtreeRebuildTopology: values.rebuildTopology !== false,
    quadtreeTallCells: {
      pressureIterations: numberValue(values, params, "pressureIterations"),
      // Narita et al. stop ICCG at a 1e-4 relative residual. The shared scene
      // default is 1e-8 for the small CPU reference and made the GPU path run
      // its entire maximum iteration budget on every step.
      relativeTolerance: Math.max(scene.numerics.pressureRelativeTolerance, 1e-4),
      adaptivityStrength: numberValue(values, params, "adaptivityStrength"),
      maximumLeafSize: typeof values.maximumLeafSize === "number" ? values.maximumLeafSize : quality === "balanced" ? 8 : 16,
      opticalDepthFraction: typeof values.opticalDepthFraction === "number" ? values.opticalDepthFraction : 0.25
    }
  }),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress) => WebGPUUniformEulerianSolver.createAsync(device, scene, quality, onRigidLoads, {
    densitySharpening: false,
    pressureIterations: numberValue(values, params, "pressureIterations"),
    tallCellSettings: { surfaceColumns: numberValue(values, params, "surfaceColumns") },
    quadtreeRebuildTopology: values.rebuildTopology !== false,
    quadtreeTallCells: {
      pressureIterations: numberValue(values, params, "pressureIterations"), relativeTolerance: Math.max(scene.numerics.pressureRelativeTolerance, 1e-4),
      adaptivityStrength: numberValue(values, params, "adaptivityStrength"),
      maximumLeafSize: typeof values.maximumLeafSize === "number" ? values.maximumLeafSize : quality === "balanced" ? 8 : 16,
      opticalDepthFraction: typeof values.opticalDepthFraction === "number" ? values.opticalDepthFraction : 0.25
    }
  }, (label, completed, total) => onProgress({ phase: label.startsWith("Building adaptive") ? "adaptive-topology" : "solver-pipelines", label, completed, total }))
};
