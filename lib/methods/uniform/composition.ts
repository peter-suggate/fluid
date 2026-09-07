import { ALGORITHM_PARAMS, algorithmFeature } from "./features/algorithms/definition";
import { parameterVariantSelections } from "../../core/method-parameter-variants";
import { composeFeatures, type FeatureDefinition } from "../../framework/composition";
import { publicationPort } from "../../framework/ports";
import type { MethodParamValues } from "../../core/method-contract";

export const surfacePublication = publicationPort<GPUTexture>({
  id:"simulation.surface", representation:"dense-surface-field-texture", lifetime:"generation",
});
const host: FeatureDefinition = {
  id:"simulation.uniform.host", provides:["simulation.dense-grid"], outputs:[surfacePublication],
  variants: [{
    id:"cm11a-lcp-multigrid", point:"simulation.uniform.pressure", requires:["simulation.dense-grid"],
    provides:["simulation.pressure-projection"], update:"rebuild", default:true,
  }, ...['dense-cm12'].map(id => ({
    id, point:"simulation.uniform.surface", requires:["simulation.dense-grid"],
    provides:["simulation.surface-publication"], update:"rebuild" as const, default:id === "dense-cm12",
  }))],
};
export function resolveMethodComposition(values: MethodParamValues = {}) {
  // Existing solver adapters test explicit strings. Boolean overrides therefore
  // retain each algorithm's default rather than becoming a new string variant.
  const algorithmValues = {...values};
  for (const param of ALGORITHM_PARAMS) {
    if (typeof algorithmValues[param.key] === "boolean") algorithmValues[param.key] = param.default;
  }
  return composeFeatures({features: [host, algorithmFeature], selections:{
    ...parameterVariantSelections("simulation.uniform.algorithms", ALGORITHM_PARAMS, algorithmValues),
    "simulation.uniform.surface": "dense-cm12",
  }});
}
