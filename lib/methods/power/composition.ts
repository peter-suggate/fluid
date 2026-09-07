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
  id:"simulation.power-liquids.host", provides:["simulation.octree"], outputs:[surfacePublication],
  variants: [{
    id:"power2017-hybrid", point:"simulation.power-liquids.pressure", requires:["simulation.octree"],
    provides:["simulation.pressure-projection"], update:"rebuild", default:true,
  }, ...['1', '4', '8'].map(id => ({
    id, point:"simulation.power-liquids.surface", requires:["simulation.octree"],
    provides:["simulation.surface-publication"], update:"rebuild" as const, default:id === "4",
  }))],
};
export function resolveMethodComposition(values: MethodParamValues = {}) {
  return composeFeatures({features: [{...host, outputs:[String(values.globalFineLevelSetFactor ?? "4") === "1" ? surfacePublication : fineSurfacePublication]}], selections:{
    "simulation.power-liquids.surface": String(values.globalFineLevelSetFactor ?? "4"),
  }});
}
