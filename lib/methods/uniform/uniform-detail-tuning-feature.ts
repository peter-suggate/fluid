import type { FeatureDefinition } from "../../framework/composition";

/**
 * Dynamic detail's tuning, as a section of the study rail.
 *
 * The method installs it only while its detail policy is Dynamic: the dials
 * write the params the census reads under that policy, and under any other
 * they would move nothing.
 */
export const uniformDetailTuningFeature = {
  id: "uniform.detail-tuning",
  label: "Detail tuning",
  controls: [
    { id: "dials", label: "Tune", kind: "number", update: "live",
      hint: "Four dials over what makes a tile run h and what it costs; All controls holds every setting on its own." },
  ],
  sections: [
    // Three sliders: the dials over the census.
    { id: "tune", surface: "study-rail", slot: "study.tune", label: "Tune", order: 10,
      hint: "Where the solver spends h cells, and how much of it.", icon: "M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1M15 4v4M9 10v4M17 16v4" },
  ],
  placements: [{ slot: "study.tune", control: "dials", presentation: "compact" }],
} as const satisfies FeatureDefinition;
