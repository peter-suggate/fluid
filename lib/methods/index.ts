import { tallCellMethod } from "./tall-cell";
import { quadtreeTallCellMethod } from "./quadtree-tall-cell";
import { octreeMethod } from "./octree";
import { uniformMethod } from "./uniform";
import { cpuReferenceMethod } from "./cpu-reference";
import type { SimulationMethod } from "./types";

export * from "./types";

/** Registry order defines picker order. Register new methods here. */
export const simulationMethods: ReadonlyArray<SimulationMethod> = [
  tallCellMethod,
  quadtreeTallCellMethod,
  octreeMethod,
  uniformMethod,
  cpuReferenceMethod
];

export const defaultMethodId = tallCellMethod.id;

export function getMethod(id: string): SimulationMethod {
  return simulationMethods.find((method) => method.id === id) ?? tallCellMethod;
}
