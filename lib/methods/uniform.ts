import { WebGPUUniformEulerianSolver } from "../webgpu-uniform-eulerian";
import { numberValue, type MethodParamSpec, type SimulationMethod } from "./types";

const params: MethodParamSpec[] = [
  { kind: "select", key: "velocityTransport", label: "Velocity transport", default: "maccormack", tier: "coarse", options: [{ value: "maccormack", label: "Bounded MacCormack" }, { value: "semi-lagrangian", label: "Semi-Lagrangian" }], hint: "Switch transport schemes while retaining the same wall, force, VOF, and pressure treatment." },
  { kind: "number", key: "jacobiIterations", label: "Jacobi iterations", unit: "iterations", min: 16, max: 400, step: 8, digits: 0, default: 64, tier: "coarse", hint: "Pressure relaxation sweeps per step. This controls incompressibility, not advection damping; deep scenes need more sweeps." },
  { kind: "number", key: "surfaceColumns", label: "Grid columns", unit: "columns", min: 1_000, max: 20_000, step: 500, digits: 0, default: 2_500, tier: "fine", hint: "Target x/z column count; the full-depth cubic grid is matched to the tall-cell layout at this density." }
];

export const uniformMethod: SimulationMethod = {
  id: "uniform",
  label: "Uniform grid VOF",
  shortLabel: "Uniform",
  badge: "UNIFORM GRID",
  description: "Full-depth cubic comparison grid with conservative VOF and selectable velocity transport.",
  detail: "the matched full-depth cubic comparison grid with conservative VOF, selectable velocity transport, and weighted Jacobi projection",
  backend: "webgpu",
  qualityLabels: { balanced: "matched cubic grid", high: "matched cubic grid", ultra: "matched cubic grid" },
  params,
  pressureMapping: "Pressure accuracy directly sets the damped-Jacobi sweep count per step.",
  presetFor: (quality) => ({
    jacobiIterations: quality === "balanced" ? 64 : quality === "high" ? 80 : 96,
    surfaceColumns: quality === "balanced" ? 2_500 : quality === "high" ? 7_000 : 12_500
  }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUUniformEulerianSolver(device, scene, quality, onRigidLoads, {
    pressureIterations: numberValue(values, params, "jacobiIterations"),
    velocityTransport: values.velocityTransport === "semi-lagrangian" ? "semi-lagrangian" : "maccormack",
    densitySharpening: values.densitySharpening !== "off",
    tallCellSettings: { surfaceColumns: numberValue(values, params, "surfaceColumns") }
  }),
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress) => WebGPUUniformEulerianSolver.createAsync(device, scene, quality, onRigidLoads, {
    pressureIterations: numberValue(values, params, "jacobiIterations"),
    velocityTransport: values.velocityTransport === "semi-lagrangian" ? "semi-lagrangian" : "maccormack",
    densitySharpening: values.densitySharpening !== "off",
    tallCellSettings: { surfaceColumns: numberValue(values, params, "surfaceColumns") }
  }, (label, completed, total) => onProgress({ phase: "solver-pipelines", label, completed, total }))
};
