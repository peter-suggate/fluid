import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type SimulationMethod } from "./types";

const params: MethodParamSpec[] = [
  { kind: "number", key: "pressureIterations", label: "Pressure iterations", unit: "iterations", min: 16, max: 1024, step: 16, digits: 0, default: 96, tier: "coarse", hint: "Hard CG safety budget; encoded work adapts to recent iterations-to-tolerance without changing the relative residual stop." },
  { kind: "number", key: "surfaceColumns", label: "Finest columns", unit: "columns", min: 1_000, max: 20_000, step: 500, digits: 0, default: 2_500, tier: "fine", hint: "Finest x/z lattice used by quadtree leaves and the cubic advection field." },
  { kind: "number", key: "adaptivityStrength", label: "Adaptivity", unit: "alpha", min: 0, max: 1, step: 0.05, digits: 2, default: 1, tier: "fine", hint: "Ando–Batty Eq. 38: 0 is the ordinary-grid limit; 1 permits full quadtree coarsening." },
  { kind: "select", key: "preconditioner", label: "Preconditioner", default: "ic0", tier: "fine", options: [{ value: "ic0", label: "Paper IC(0)" }, { value: "line", label: "Vertical line" }, { value: "poly", label: "Polynomial" }, { value: "jacobi", label: "Parallel Jacobi" }], hint: "IC(0) is the paper reference; line and polynomial paths preserve the operator and tolerance while avoiding serial triangular levels." }
];

const preconditionerValue = (value: unknown) => value === "line" || value === "poly" || value === "jacobi" ? value : "ic0";

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
  presetFor: (quality) => ({ pressureIterations: quality === "balanced" ? 96 : quality === "high" ? 160 : 240, surfaceColumns: quality === "balanced" ? 2_500 : quality === "high" ? 7_000 : 12_500, adaptivityStrength: 1, preconditioner: "ic0" }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUUniformEulerianSolver(device, scene, quality, onRigidLoads, {
    // Narita Sec. 4.5 advects the level set from the saved previous grid.
    // It is authoritative for adaptive pressure geometry, while the shared
    // cubic backing transport runs without density sharpening. Velocity uses
    // bounded MacCormack like the uniform and restricted references —
    // first-order SL transport measurably damped the dam-break collapse
    // (peak speed 4 vs 13 m/s against the restricted method).
    densitySharpening: false,
    velocityTransport: "maccormack",
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
      opticalDepthFraction: typeof values.opticalDepthFraction === "number" ? values.opticalDepthFraction : 0.25,
      preconditioner: preconditionerValue(values.preconditioner),
      polynomialDegree: typeof values.polynomialDegree === "number" ? values.polynomialDegree : 2,
      debugPressureTimings: values.debugPressureTimings === true
    }
  }),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress) => WebGPUUniformEulerianSolver.createAsync(device, scene, quality, onRigidLoads, {
    densitySharpening: false,
    velocityTransport: "maccormack",
    pressureIterations: numberValue(values, params, "pressureIterations"),
    tallCellSettings: { surfaceColumns: numberValue(values, params, "surfaceColumns") },
    quadtreeRebuildTopology: values.rebuildTopology !== false,
    quadtreeTallCells: {
      pressureIterations: numberValue(values, params, "pressureIterations"), relativeTolerance: Math.max(scene.numerics.pressureRelativeTolerance, 1e-4),
      adaptivityStrength: numberValue(values, params, "adaptivityStrength"),
      maximumLeafSize: typeof values.maximumLeafSize === "number" ? values.maximumLeafSize : quality === "balanced" ? 8 : 16,
      opticalDepthFraction: typeof values.opticalDepthFraction === "number" ? values.opticalDepthFraction : 0.25,
      preconditioner: preconditionerValue(values.preconditioner),
      polynomialDegree: typeof values.polynomialDegree === "number" ? values.polynomialDegree : 2,
      debugPressureTimings: values.debugPressureTimings === true
    }
  }, (label, completed, total) => onProgress({ phase: label.startsWith("Building adaptive") ? "adaptive-topology" : "solver-pipelines", label, completed, total }))
};
