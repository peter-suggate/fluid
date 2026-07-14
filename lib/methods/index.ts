import { tallCellMethod } from "./tall-cell";
import { adaptiveOpticalLayerMethod } from "./adaptive-optical-layer";
import { uniformMethod } from "./uniform";
import { cpuReferenceMethod } from "./cpu-reference";
import type { SimulationMethod } from "./types";

export * from "./types";

/** Registry order defines picker order. Register new methods here. */
export const simulationMethods: ReadonlyArray<SimulationMethod> = [
  tallCellMethod,
  adaptiveOpticalLayerMethod,
  uniformMethod,
  cpuReferenceMethod
];

export const defaultMethodId = tallCellMethod.id;

export function getMethod(id: string): SimulationMethod {
  return simulationMethods.find((method) => method.id === id) ?? tallCellMethod;
}
