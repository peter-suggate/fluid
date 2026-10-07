import { numberValue, type MethodParamSpec, type MethodParamValues } from "../../../core/method-contract";
import type { SceneDescription } from "../../../core/model";

export const MAC_PARAMS: MethodParamSpec[] = [
  { kind: "number", key: "maxStep", label: "Maximum step", default: 1 / 120, min: 0.0001, max: 1 / 30, step: 0.0001, digits: 4, unit: "s", tier: "coarse", update: "runtime", hint: "GPU substeps respect velocity, gravity, viscosity and surface tension. Reduce this limit if the GPU continuation capacity is exhausted." },
  { kind: "number", key: "cfl", label: "Advection CFL", default: 0.5, min: 0.1, max: 1, step: 0.1, unit: "cells", tier: "coarse", update: "runtime" },
  { kind: "select", key: "advection", label: "Advection", default: "maccormack", options: [{ value: "maccormack", label: "Bounded MacCormack" }, { value: "semi-lagrangian", label: "Semi-Lagrangian" }], tier: "coarse", update: "runtime" },
  { kind: "number", key: "pressureTolerance", label: "Divergence tolerance", default: 0.001, min: 0.00001, max: 0.1, step: 0.00001, digits: 5, unit: "s⁻¹", tier: "fine", update: "runtime", hint: "Fresh b − Ap infinity norm. A step is accepted only after meeting this bound." },
  { kind: "number", key: "pressureLimit", label: "Pressure iteration limit", default: 256, min: 32, max: 2048, step: 32, unit: "iterations", tier: "fine", update: "runtime", hint: "A safety limit, not an accuracy control: exhaustion stops the simulation." },
];

export function macOptions(values: MethodParamValues = {}) {
  return {
    maxStep: numberValue(values, MAC_PARAMS, "maxStep"),
    cfl: numberValue(values, MAC_PARAMS, "cfl"),
    maccormack: values.advection !== "semi-lagrangian",
    tolerance: numberValue(values, MAC_PARAMS, "pressureTolerance"),
    pressureLimit: Math.round(numberValue(values, MAC_PARAMS, "pressureLimit")),
  };
}

/** Acceleration-aware displacement bound; zero velocity at release is not an infinite step. */
export function macTimeStep(scene: SceneDescription, h: readonly number[], maxSpeed: number, values: ReturnType<typeof macOptions>, remaining: number): number {
  const dx = Math.min(...h), distance = values.cfl * dx;
  const g = scene.fluid.gravity_m_s2;
  const acceleration = Math.hypot(g.x, g.y, g.z);
  const advective = 2 * distance / Math.max(maxSpeed + Math.sqrt(maxSpeed ** 2 + 2 * acceleration * distance), 1e-30);
  const nu = scene.fluid.dynamicViscosity_Pa_s / scene.fluid.density_kg_m3;
  const viscous = nu > 0 ? 0.45 / (nu * h.reduce((sum, x) => sum + 1 / (x * x), 0)) : Infinity;
  const sigma = scene.fluid.surfaceTension_N_m;
  const capillary = sigma > 0 ? 0.5 * Math.sqrt(scene.fluid.density_kg_m3 * dx ** 3 / (Math.PI * sigma)) : Infinity;
  return Math.min(remaining, values.maxStep, advective, viscous, capillary);
}

export function validateMacScene(scene: SceneDescription): void {
  if (scene.rigidBodies.length) throw new Error("Uniform MAC baseline currently supports fixed voxel solids; choose a scene without rigid bodies.");
  if (scene.fluid.inflow && scene.fluid.inflow.enabled !== false)
    throw new Error("Uniform MAC baseline currently supports closed liquid inventories; disable the inflow for this comparison.");
}
