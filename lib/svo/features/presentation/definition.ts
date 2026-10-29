import type { FeatureDefinition } from "../../../framework/composition";
export const SVO_PRESENTATION_FEATURE = {
  id: "svo.presentation",
  controls: [{ id: "resolution", label: "Render resolution", kind: "number", setting: "resolutionScale",
    min: .35, max: 2, step: .01, update: "live" }],
  placements: [{ slot: "view.performance", control: "resolution", presentation: "compact" }],
} as const satisfies FeatureDefinition;
