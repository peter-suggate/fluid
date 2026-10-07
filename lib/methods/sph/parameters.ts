import { numberValue, type MethodParamSpec, type MethodParamValues } from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";

export const SPH_PARAMS: MethodParamSpec[] = [
  { kind: "select", key: "particlesPerCell", label: "Particle sampling", default: "1", options: [{ value: "1", label: "1 per cell" }, { value: "8", label: "8 per cell" }], tier: "coarse", update: "solver", hint: "One particle per wet cell matches the scene lattice. Eight doubles particle resolution in each axis, costs eight times the particles and halves the acoustic timestep." },
  { kind: "number", key: "artificialViscosity", label: "Acoustic damping", default: 0.2, min: 0, max: 1, step: 0.01, digits: 2, unit: "α", tier: "fine", update: "runtime", hint: "Traditional approaching-pair artificial viscosity damps compressive shocks. This numerical damping is separate from the scene's physical viscosity." },
  { kind: "number", key: "soundSpeed", label: "Pressure stiffness (sound speed)", default: 20, min: 1, max: 100, step: 1, unit: "m/s", tier: "coarse", update: "runtime", hint: "p = c² max(density − rest density, 0). Higher values reduce compression and require smaller substeps." },
  { kind: "number", key: "cfl", label: "SPH timestep safety", default: 0.2, min: 0.05, max: 0.3, step: 0.01, digits: 2, unit: "", tier: "coarse", update: "runtime", hint: "GPU substeps are limited by acoustic speed, particle motion, acceleration, viscosity and capillarity." },
];
export const sphOptions = (values: MethodParamValues = {}) => ({ artificialViscosity: numberValue(values, SPH_PARAMS, "artificialViscosity"), soundSpeed: numberValue(values, SPH_PARAMS, "soundSpeed"), cfl: numberValue(values, SPH_PARAMS, "cfl") });
export function validateSphScene(scene: SceneDescription): void {
  const f = scene.fluid;
  if (!(f.density_kg_m3 > 0) || !Number.isFinite(f.density_kg_m3) || ![f.dynamicViscosity_Pa_s, f.surfaceTension_N_m].every(v => Number.isFinite(v) && v >= 0)
    || !Object.values(f.gravity_m_s2).every(Number.isFinite) || (f.initialVelocity_m_s && !Object.values(f.initialVelocity_m_s).every(Number.isFinite)))
    throw new Error("SPH requires finite fluid properties, positive density and nonnegative viscosity/surface tension.");
}
/** Remove the small quadrature bias of the fixed seed lattice.
 * This is a constant kernel normalization, never a correction of live density. */
export function sphKernelNormalization(h: readonly number[], samplesPerAxis = 2): number {
  const radius = 2 * Math.min(...h) / samplesPerAxis, volume = h[0] * h[1] * h[2] / samplesPerAxis ** 3;
  let sum = 0;
  for (let z = -2; z <= 2; z++) for (let y = -2; y <= 2; y++) for (let x = -2; x <= 2; x++) {
    const q = ((x * h[0] / samplesPerAxis) ** 2 + (y * h[1] / samplesPerAxis) ** 2 + (z * h[2] / samplesPerAxis) ** 2) / radius ** 2;
    sum += volume * 315 / (64 * Math.PI * radius ** 3) * Math.max(0, 1 - q) ** 3;
  }
  return 1 / sum;
}
