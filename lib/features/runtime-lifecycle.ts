import { topologyFreezeLifecycle, topologyFreezeQuery, type TopologyFreezeState } from "./topology-freeze/definition";
import { combineQueryCodecs } from "../framework/persistence";
const features = [topologyFreezeLifecycle];
export const runtimeFeatureQuery = combineQueryCodecs<TopologyFreezeState>([topologyFreezeQuery]);
export function initialRuntimeFeatures(): TopologyFreezeState {
  return Object.assign({}, ...features.map(feature => feature.initial()));
}
export function resetRuntimeFeatures(event: "simulation"): Partial<TopologyFreezeState> {
  return Object.assign({}, ...features.filter(feature => feature.reset === event).map(feature => feature.initial()));
}
