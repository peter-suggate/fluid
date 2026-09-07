import { ALGORITHM_PARAMS, algorithmFeature } from "./features/algorithms/definition";
import { parameterVariantSelections } from "../../core/method-parameter-variants";
import { composeFeatures, type FeatureDefinition } from "../../framework/composition";
import { publicationPort } from "../../framework/ports";
import type { MethodParamValues } from "../../core/method-contract";
import type { CoarseLevelSetConsumerSource, WebGPUFineLevelSetBrickSource } from "../../core/levelset-consumer-abi";

export const surfacePublication = publicationPort<CoarseLevelSetConsumerSource>({
  id:"simulation.surface", representation:"octree", lifetime:"generation",
});
export const fineSurfacePublication = publicationPort<WebGPUFineLevelSetBrickSource>({
  id:"simulation.surface", representation:"sparse-fine-levelset-bricks", lifetime:"generation",
});
const host: FeatureDefinition = {
  id:"simulation.losasso.host", provides:["simulation.octree"], outputs:[surfacePublication],
  variants: [{
    id:"vcycle-mgpcg", point:"simulation.losasso.pressure", requires:["simulation.octree"],
    provides:["simulation.pressure-projection"], update:"rebuild", default:true,
  }, ...['1', '4', '8'].map(id => ({
    id, point:"simulation.losasso.surface", requires:["simulation.octree"],
    provides:["simulation.surface-publication"], update:"rebuild" as const, default:id === "1",
  }))],
};
export function resolveMethodComposition(values: MethodParamValues = {}) {
  return composeFeatures({features: [{...host, outputs:[String(values.globalFineLevelSetFactor ?? "1") === "1" ? surfacePublication : fineSurfacePublication]}, algorithmFeature], selections:{
    ...parameterVariantSelections("simulation.losasso.algorithms", ALGORITHM_PARAMS, values),
    "simulation.losasso.surface": String(values.globalFineLevelSetFactor ?? "1"),
  }});
}
