import { octreeMethod } from "./octree";
import { uniformMethod } from "./uniform";
import { cpuReferenceMethod } from "./cpu-reference";
import type { SimulationMethod } from "./types";

export * from "./types";
export * from "../octree-coarse-backend";

/** Complete registry used by runtime lookup and offline comparison tooling. */
export const simulationMethods: ReadonlyArray<SimulationMethod> = [
  octreeMethod,
  uniformMethod,
  cpuReferenceMethod
];

// Uniform is the default for any scene that does not author its own method
// profile. Profiled presets (the shipped catalog's octree scenes among them)
// are untouched: `profile?.methodId` wins over this everywhere it is read.
export const defaultMethodId = uniformMethod.id;

/** Methods supported as interactive production/experimental choices. The
 * broader registry remains available to offline comparison tooling. */
export const interactiveSimulationMethods: ReadonlyArray<SimulationMethod> = [
  octreeMethod,
  uniformMethod
];

export function interactiveMethodId(id: string): string {
  return interactiveSimulationMethods.some((method) => method.id === id) ? id : defaultMethodId;
}

export function getMethod(id: string): SimulationMethod {
  return simulationMethods.find((method) => method.id === id) ?? uniformMethod;
}
