import type { FeatureDefinition } from "../../framework/composition";

/**
 * The view rail's own sections: topics that belong to no one feature.
 *
 * A section a feature owns outright — Lighting — is declared by that feature.
 * These are the shared ones, which any installed feature may place controls
 * in; the rail draws one only while something has.
 */
export const viewRailFeature = {
  id: "view.rail",
  label: "View rail",
  sections: [
    // A gauge: what the renderer may skip, and when.
    { id: "performance", surface: "view-rail", slot: "view.performance", label: "Performance", order: 100,
      hint: "What the renderer may skip, and when.", icon: "M12 14l4-4M3.34 19a10 10 0 1 1 17.32 0" },
  ],
} as const satisfies FeatureDefinition;
