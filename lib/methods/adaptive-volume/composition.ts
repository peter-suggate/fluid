import { pressureInspectionFeature } from "../../features/pressure-inspection/definition";
import { topologyFreezeFeature } from "../../features/topology-freeze/definition";
import { ALGORITHM_PARAMS, algorithmFeature } from "./features/algorithms/definition";
import { parameterVariantSelections } from "../../core/method-parameter-variants";
import { composeFeatures, type FeatureDefinition } from "../../framework/composition";
import { publicationPort } from "../../framework/ports";
import type { MethodParamValues } from "../../core/method-contract";
import { adaptiveMassAdaptivityFeature } from "./features/adaptivity/definition";
import type { SparseCM12FinePresentationSource } from "./webgpu-sparse-cm12-resident";

export const surfacePublication = publicationPort<SparseCM12FinePresentationSource>({
  id:"simulation.surface", representation:"sparse-atlas", lifetime:"generation",
});
const host: FeatureDefinition = {
  id:"simulation.adaptive-volume.host", provides:["simulation.sparse-atlas", "simulation.pressure-journal"], outputs:[surfacePublication],
  variants: [{
    id:"sparse-jacobi-pcg", point:"simulation.adaptive-volume.pressure", requires:["simulation.sparse-atlas"],
    provides:["simulation.pressure-projection"], update:"rebuild", default:true,
  }, ...['sparse-cm12'].map(id => ({
    id, point:"simulation.adaptive-volume.surface", requires:["simulation.sparse-atlas"],
    provides:["simulation.surface-publication"], update:"rebuild" as const, default:id === "sparse-cm12",
  }))],
};
export function resolveMethodComposition(values: MethodParamValues = {}) {
  // Existing solver adapters test explicit strings. Boolean overrides therefore
  // retain each algorithm's default rather than becoming a new string variant.
  const algorithmValues = {...values};
  for (const param of ALGORITHM_PARAMS) {
    if (typeof algorithmValues[param.key] === "boolean") algorithmValues[param.key] = param.default;
  }
  return composeFeatures({features: [host, adaptiveMassAdaptivityFeature, algorithmFeature, topologyFreezeFeature, pressureInspectionFeature], selections:{
    ...parameterVariantSelections("simulation.adaptive-volume.algorithms", ALGORITHM_PARAMS, algorithmValues),
    "simulation.adaptive-volume.surface": "sparse-cm12",
    "simulation.adaptive-volume.adaptivity": String(values.selectorMode ?? "coarse-first"),
  }});
}
