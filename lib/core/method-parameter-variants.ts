import type { MethodParamSpec, MethodParamValues } from "./method-contract";
import type { FeatureDefinition } from "../framework/composition";

/** Adapt a method-owned choice schema; numeric tuning and diagnostic controls are never inferred. */
export function parameterVariantFeature(
  id: string, params: readonly MethodParamSpec[], requires: readonly string[],
): FeatureDefinition {
  return {
    id, requires,
    variants: params.flatMap(param => {
      if (param.kind !== "select") throw new Error(`Algorithm variant ${id}/${param.key} must be a choice`);
      return param.options.map(option => ({
        id:option.value, label:option.label, point:`${id}.${param.key}`,
        requires, provides:[`${id}.${param.key}.selected`],
        update:param.update === "runtime" ? "live" as const : "rebuild" as const,
        default:option.value === param.default,
      }));
    }),
  };
}
export function parameterVariantSelections(id: string, params: readonly MethodParamSpec[], values: MethodParamValues) {
  return Object.fromEntries(params.map(param => [`${id}.${param.key}`, String(values[param.key] ?? param.default)]));
}
