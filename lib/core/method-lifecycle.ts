import { resolveMethodValues, type SimulationMethod, type MethodParamValues } from "./method-contract";
import type { GPUQuality } from "./gpu-quality";
import { configurationChangeImpact } from "../framework/lifecycle";

/** Translate the method schema into the shared lifecycle vocabulary once. */
export function methodConfigurationImpact(method: SimulationMethod, quality: GPUQuality,
  before: MethodParamValues, after: MethodParamValues) {
  const previous = resolveMethodValues(method, quality, before);
  const next = resolveMethodValues(method, quality, after);
  return configurationChangeImpact(method.resolveComposition(previous), method.resolveComposition(next), previous, next,
    method.params.map(param => ({ key: param.key, impact: param.update === "runtime" ? "live" : "rebuild" })));
}
