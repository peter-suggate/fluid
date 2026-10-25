import type { FeatureDefinition } from "../../framework/composition";

/**
 * The composed visual layers, as a section of the study rail.
 *
 * Installed only for a method that publishes its field views as layers
 * (`capabilities.visualLayers`). The card holds the layers and their plane;
 * the legend is placed apart so the rail keeps a drawn layer's colour key in
 * view while every card is closed.
 */
export const visualLayersFeature = {
  id: "presentation.visual-layers",
  label: "Visual layers",
  controls: [
    { id: "layers", label: "Layers", kind: "choice",
      hint: "Combine any of the layers the solver publishes, on a slice plane or through the volume." },
    { id: "legend", label: "Layer key", kind: "readout" },
  ],
  sections: [
    // Three stacked sheets: what is drawn over the picture, and in what order.
    { id: "layers", surface: "study-rail", slot: "study.layers", label: "Layers", order: 20,
      hint: "What the solver's fields draw over the scene.", icon: "M12 2 2 7l10 5 10-5ZM2 12l10 5 10-5M2 17l10 5 10-5" },
  ],
  placements: [
    { slot: "study.layers", control: "layers", presentation: "compact" },
    { slot: "rail.legend", control: "legend", presentation: "compact" },
  ],
} as const satisfies FeatureDefinition;
