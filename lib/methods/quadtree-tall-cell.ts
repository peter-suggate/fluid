import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { quadtreeMegakernelDofLimit, quadtreeMegakernelRowIterationLimit, type QuadtreeMegakernelMode, type QuadtreePressureSolver } from "../webgpu-quadtree-tall-cell";
import { numberValue, type MethodParamSpec, type SimulationMethod } from "./types";

const params: MethodParamSpec[] = [
  { kind: "number", key: "pressureIterations", label: "Pressure iterations", unit: "iterations", min: 16, max: 1024, step: 16, digits: 0, default: 96, tier: "coarse", hint: "Minimum PCG safety budget. The experimental Chebyshev solver uses the same number of fixed row-parallel passes." },
  { kind: "number", key: "adaptivityStrength", label: "Adaptivity", unit: "alpha", min: 0, max: 1, step: 0.05, digits: 2, default: 1, tier: "fine", hint: "Ando–Batty Eq. 38: 0 is the ordinary-grid limit; 1 permits full quadtree coarsening." },
  { kind: "select", key: "opticalLayerMode", label: "Optical layer", default: "adaptive-motion", tier: "coarse", options: [{ value: "adaptive-motion", label: "Motion-adaptive (2026)" }, { value: "fixed", label: "Fixed quarter-depth" }], hint: "Narita–Kanai 2026 derives a smooth per-column layer thickness from tall-cell velocity reconstruction error. Fixed retains the previous baseline for A/B measurements." },
  { kind: "number", key: "opticalAlpha", label: "Optical motion response", unit: "alpha", min: 0, max: 2, step: 0.05, digits: 2, default: 0.5, tier: "fine", hint: "Paper Eq. (1) scale for motion-sensitive dilation. Larger values retain more cubic pressure cells around dynamic flow." },
  { kind: "select", key: "pressureSolver", label: "Pressure solver", default: "pcg", tier: "fine", options: [{ value: "pcg", label: "PCG (stable default)" }, { value: "chebyshev", label: "Chebyshev (experimental)" }], hint: "PCG converges to the configured tolerance and retains exact same-step low-rank rigid response. Chebyshev removes Krylov reductions but uses fixed work and frame-lagged rigid exchange." },
  { kind: "select", key: "preconditioner", label: "Preconditioner", default: "poly", tier: "fine", options: [{ value: "mg", label: "Geometric multigrid" }, { value: "poly", label: "Polynomial" }, { value: "ic0", label: "Paper IC(0)" }, { value: "blockic", label: "Block IC(0)" }, { value: "line", label: "Vertical line" }, { value: "jacobi", label: "Parallel Jacobi" }], hint: "Geometric multigrid uses dyadic quadtree aggregation, Galerkin coarse operators, and vertical block smoothing. Polynomial remains the measured default until the multigrid benchmark gate passes." },
  { kind: "select", key: "megakernelMode", label: "CG megakernel", default: "dynamic", tier: "fine", options: [{ value: "dynamic", label: "Dynamic (recommended)" }, { value: "always", label: "Forced on" }, { value: "off", label: "Off" }], hint: "Dynamic uses the previous converged solve to choose between one persistent workgroup and the parallel dispatch ladder. Forced on applies only to supported uncoupled Polynomial/Jacobi solves." },
  { kind: "number", key: "megakernelDofLimit", label: "Megakernel DOF limit", unit: "DOFs", min: 256, max: 131_072, step: 256, digits: 0, default: quadtreeMegakernelDofLimit, tier: "fine", hint: "Dynamic mode uses the dispatch ladder above this pressure-system size. Forced on ignores this limit." },
  { kind: "number", key: "megakernelRowIterationLimit", label: "Megakernel work limit", unit: "row-iterations", min: 1_000, max: 500_000, step: 1_000, digits: 0, default: quadtreeMegakernelRowIterationLimit, tier: "fine", hint: "Dynamic mode requires DOFs × previous iterations × polynomial cost at or below this value. Forced on ignores this limit." },
  { kind: "select", key: "vofReconciliation", label: "Emergency VOF recovery", default: "on", tier: "fine", options: [{ value: "on", label: "Armed" }, { value: "off", label: "Strict φ-only" }], hint: "Leaves healthy level-set transport untouched. If represented liquid falls below -10%, conservative VOF may restore lost liquid until recovery reaches -2%." }
];

const preconditionerValue = (value: unknown) => value === "ic0" || value === "blockic" || value === "line" || value === "jacobi" || value === "mg" ? value : "poly";
const opticalLayerModeValue = (value: unknown) => value === "fixed" ? "fixed" as const : "adaptive-motion" as const;
const megakernelModeValue = (values: Record<string, unknown>): QuadtreeMegakernelMode => {
  if (values.megakernelSolve === false || values.megakernelSolve === "off") return "off";
  return values.megakernelMode === "always" || values.megakernelMode === "off" ? values.megakernelMode : "dynamic";
};
const pressureSolverValue = (value: unknown): QuadtreePressureSolver => value === "chebyshev" ? "chebyshev" : "pcg";

export const quadtreeTallCellMethod: SimulationMethod = {
  id: "quadtree-tall-cell",
  label: "Quadtree tall cells",
  shortLabel: "Adaptive",
  badge: "QUADTREE TALL CELLS",
  description: "Narita et al. 2025 quadtree tall cells with Narita–Kanai 2026 motion-adaptive optical layers.",
  detail: "motion-error optical thickness with constrained smoothing, coarse-to-fine quadtree sizing, strict 2:1 adaptivity, multiple vertical tall runs, horizontally centered pressure samples, T-junction face-overlap gradients, corrected inner ghost volumes, SPD free-surface scaling, and a tolerance-driven PCG pressure solve",
  backend: "webgpu",
  qualityLabels: { balanced: "bounded pressure work", high: "higher pressure work", ultra: "maximum pressure work" },
  params,
  pressureMapping: "Balanced starts PCG with a 96-iteration safety budget and stops at the relative residual tolerance. Experimental Chebyshev uses 96 fixed passes.",
  presetFor: (quality) => ({ pressureIterations: quality === "balanced" ? 96 : quality === "high" ? 160 : 240, adaptivityStrength: 1, opticalLayerMode: "adaptive-motion", opticalAlpha: 0.5, pressureSolver: "pcg", preconditioner: "poly" }),
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
    quadtreeRebuildTopology: values.rebuildTopology !== false,
    ...(typeof values.topologyStaleSteps === "number" ? { quadtreeTopologyStaleSteps: values.topologyStaleSteps } : {}),
    ...(typeof values.inlineRebuild === "boolean" ? { quadtreeInlineRebuild: values.inlineRebuild } : {}),
    quadtreeTallCells: {
      pressureIterations: numberValue(values, params, "pressureIterations"),
      pressureSolver: pressureSolverValue(values.pressureSolver),
      // Narita et al. stop ICCG at a 1e-4 relative residual. The shared scene
      // default is 1e-8 for the small CPU reference and made the GPU path run
      // its entire maximum iteration budget on every step.
      relativeTolerance: Math.max(scene.numerics.pressureRelativeTolerance, 1e-4),
      adaptivityStrength: numberValue(values, params, "adaptivityStrength"),
      maximumLeafSize: typeof values.maximumLeafSize === "number" ? values.maximumLeafSize : quality === "balanced" ? 8 : 16,
      opticalDepthFraction: typeof values.opticalDepthFraction === "number" ? values.opticalDepthFraction : 0.25,
      opticalLayerMode: opticalLayerModeValue(values.opticalLayerMode),
      opticalAlpha: numberValue(values, params, "opticalAlpha"),
      deepSpeedGradientScale: typeof values.deepSpeedGradientScale === "number" ? values.deepSpeedGradientScale : 1,
      pressureWarmStart: values.pressureWarmStart !== "off" && values.pressureWarmStart !== false,
      megakernelMode: megakernelModeValue(values),
      megakernelDofLimit: numberValue(values, params, "megakernelDofLimit"),
      megakernelRowIterationLimit: numberValue(values, params, "megakernelRowIterationLimit"),
      preconditioner: preconditionerValue(values.preconditioner),
      polynomialDegree: typeof values.polynomialDegree === "number" ? values.polynomialDegree : 2,
      vofReconciliation: values.vofReconciliation !== "off" && values.vofReconciliation !== false,
      debrisCulling: values.debrisCulling === true,
      debugPressureTimings: values.debugPressureTimings === true
    }
  }),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress) => WebGPUUniformEulerianSolver.createAsync(device, scene, quality, onRigidLoads, {
    densitySharpening: false,
    velocityTransport: "maccormack",
    pressureIterations: numberValue(values, params, "pressureIterations"),
    quadtreeRebuildTopology: values.rebuildTopology !== false,
    ...(typeof values.topologyStaleSteps === "number" ? { quadtreeTopologyStaleSteps: values.topologyStaleSteps } : {}),
    ...(typeof values.inlineRebuild === "boolean" ? { quadtreeInlineRebuild: values.inlineRebuild } : {}),
    quadtreeTallCells: {
      pressureIterations: numberValue(values, params, "pressureIterations"), relativeTolerance: Math.max(scene.numerics.pressureRelativeTolerance, 1e-4),
      pressureSolver: pressureSolverValue(values.pressureSolver),
      adaptivityStrength: numberValue(values, params, "adaptivityStrength"),
      maximumLeafSize: typeof values.maximumLeafSize === "number" ? values.maximumLeafSize : quality === "balanced" ? 8 : 16,
      opticalDepthFraction: typeof values.opticalDepthFraction === "number" ? values.opticalDepthFraction : 0.25,
      opticalLayerMode: opticalLayerModeValue(values.opticalLayerMode),
      opticalAlpha: numberValue(values, params, "opticalAlpha"),
      deepSpeedGradientScale: typeof values.deepSpeedGradientScale === "number" ? values.deepSpeedGradientScale : 1,
      pressureWarmStart: values.pressureWarmStart !== "off" && values.pressureWarmStart !== false,
      megakernelMode: megakernelModeValue(values),
      megakernelDofLimit: numberValue(values, params, "megakernelDofLimit"),
      megakernelRowIterationLimit: numberValue(values, params, "megakernelRowIterationLimit"),
      preconditioner: preconditionerValue(values.preconditioner),
      polynomialDegree: typeof values.polynomialDegree === "number" ? values.polynomialDegree : 2,
      vofReconciliation: values.vofReconciliation !== "off" && values.vofReconciliation !== false,
      debrisCulling: values.debrisCulling === true,
      debugPressureTimings: values.debugPressureTimings === true
    }
  }, (label, completed, total) => onProgress({ phase: label.startsWith("Building adaptive") ? "adaptive-topology" : "solver-pipelines", label, completed, total }))
};
