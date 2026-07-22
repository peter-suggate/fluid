import { tallCellMethod } from "./tall-cell";
import { quadtreeTallCellMethod } from "./quadtree-tall-cell";
import { octreeMethod } from "./octree";
import { uniformMethod } from "./uniform";
import { cpuReferenceMethod } from "./cpu-reference";
import type { SimulationMethod } from "./types";

export * from "./types";

/** Complete registry used by runtime lookup and offline comparison tooling. */
export const simulationMethods: ReadonlyArray<SimulationMethod> = [
  tallCellMethod,
  quadtreeTallCellMethod,
  octreeMethod,
  uniformMethod,
  cpuReferenceMethod
];

export const defaultMethodId = octreeMethod.id;

/** Methods supported as interactive production/experimental choices. The
 * broader registry remains available to offline comparison tooling. */
export const interactiveSimulationMethods: ReadonlyArray<SimulationMethod> = [
  octreeMethod,
  tallCellMethod
];

export function interactiveMethodId(id: string): string {
  return interactiveSimulationMethods.some((method) => method.id === id) ? id : defaultMethodId;
}

export function getMethod(id: string): SimulationMethod {
  return simulationMethods.find((method) => method.id === id) ?? octreeMethod;
}
