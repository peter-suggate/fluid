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
  controls: [
    { id: "load", label: "Machine load", kind: "readout",
      hint: "What the worker thread, the GPU queue, the page's thread and the CPU are doing." },
    { id: "instruments", label: "Instruments", kind: "toggle",
      hint: "Readings of the whole run: the advance pipeline, the frame pipeline and the metric cards." },
  ],
  sections: [
    // A gauge: what the frame costs, what the renderer may skip, and the instruments that explain both.
    { id: "performance", surface: "view-rail", slot: "view.performance", label: "Performance", order: 100,
      hint: "What the frame costs, and what the renderer may skip.", icon: "M12 14l4-4M3.34 19a10 10 0 1 1 17.32 0" },
  ],
  placements: [
    { slot: "view.performance", control: "load", presentation: "compact", priority: "high" },
    { slot: "view.performance", control: "instruments", presentation: "compact" },
  ],
} as const satisfies FeatureDefinition;
