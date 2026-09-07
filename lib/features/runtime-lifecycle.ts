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

export type RuntimeFeatureState = TopologyFreezeState;
export function pickRuntimeFeatures(state: RuntimeFeatureState): RuntimeFeatureState {
  return Object.fromEntries(Object.keys(initialRuntimeFeatures()).map(key => [key, state[key as keyof RuntimeFeatureState]])) as unknown as RuntimeFeatureState;
}
export function runtimeFeaturesChanged(before: RuntimeFeatureState, after: RuntimeFeatureState): boolean {
  return Object.keys(initialRuntimeFeatures()).some(key => !Object.is(before[key as keyof RuntimeFeatureState], after[key as keyof RuntimeFeatureState]));
}
