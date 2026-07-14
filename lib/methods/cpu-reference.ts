import type { MethodParamSpec, SimulationMethod } from "./types";

const params: MethodParamSpec[] = [
  { kind: "number", key: "cellSize_m", label: "Grid cell size", unit: "m", min: 0.0125, max: 0.08, step: 0.0025, digits: 4, default: 0.02, tier: "coarse", hint: "MAC grid resolution for the reference solve. Smaller cells raise fidelity and cost cubically." }
];

export const cpuReferenceMethod: SimulationMethod = {
  id: "cpu-reference",
  label: "CPU reference (binary64)",
  shortLabel: "CPU ref",
  badge: "CPU ORACLE",
  description: "Double-precision MAC validation oracle with PCG projection. Slow but exact.",
  detail: "staggered MAC, RK2 semi-Lagrangian advection, explicit viscosity, marker free surface, closed-wall flux enforcement, and matrix-free Jacobi-PCG projection",
  backend: "cpu",
  qualityLabels: { balanced: "~110k cells · 0.02 m", high: "~215k cells · 0.016 m", ultra: "~450k cells · 0.0125 m" },
  params,
  pressureMapping: "Uses the scene numerics directly: PCG runs to the relative tolerance, capped by the iteration budget.",
  // Matches the equivalent cell counts of the WebGPU balanced/high/ultra
  // presets so the reference is comparable in fidelity, even though it is
  // far slower per step.
  presetFor: (quality) => ({ cellSize_m: quality === "balanced" ? 0.02 : quality === "high" ? 0.016 : 0.0125 })
};
