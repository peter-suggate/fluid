import { octreeMethod } from "./octree";
import { cpuReferenceMethod } from "./cpu-reference";
import type { SimulationMethod } from "./types";

export * from "./types";
export * from "../octree-coarse-backend";

/** Complete registry used by runtime lookup and offline comparison tooling. */
export const simulationMethods: ReadonlyArray<SimulationMethod> = [
  octreeMethod,
  cpuReferenceMethod
];

export const defaultMethodId = octreeMethod.id;

/** Methods supported as interactive production/experimental choices. The
 * broader registry remains available to offline comparison tooling. */
export const interactiveSimulationMethods: ReadonlyArray<SimulationMethod> = [
  octreeMethod
];

export function interactiveMethodId(id: string): string {
  return interactiveSimulationMethods.some((method) => method.id === id) ? id : defaultMethodId;
}

export function getMethod(id: string): SimulationMethod {
  return simulationMethods.find((method) => method.id === id) ?? octreeMethod;
}
