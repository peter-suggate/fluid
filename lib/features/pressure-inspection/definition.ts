import type { FeatureDefinition } from "../../framework/composition";
export const pressureInspectionFeature: FeatureDefinition = {
  id: "simulation.pressure-inspection",
  requires: ["simulation.pressure-journal"],
  controls: [{ id: "film", label: "Captured solve", kind: "action" }],
  placements: [{ control: "film", slot: "fluid.inspection" }, { control: "film", slot: "sim.inspection" }],
};
