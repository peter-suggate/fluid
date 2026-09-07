import type { FeatureDefinition } from "../../framework/composition";
import { booleanQuery, queryRecord } from "../../framework/persistence";
export interface TopologyFreezeState { topologyFrozen: boolean }
export const topologyFreezeQuery = queryRecord<TopologyFreezeState>({ topologyFrozen: booleanQuery("freezeTopology", false) });
export const topologyFreezeLifecycle = {
  initial: (): TopologyFreezeState => topologyFreezeQuery.read(new URLSearchParams()),
  reset: "simulation" as const,
};
export const topologyFreezeFeature = {
  id: "simulation.topology-freeze", requires: ["simulation.sparse-atlas"],
  controls: [{ id: "enabled", label: "Freeze topology", kind: "toggle", setting: "topologyFrozen", update: "live",
    hint: "Hold existing brick coarseness. New fluid support can still grow as water moves." }],
  placements: [{ slot: "scene.simulation", control: "enabled", presentation: "compact" },
    { slot: "sim.topology", control: "enabled", presentation: "expanded" }],
} as const satisfies FeatureDefinition;
