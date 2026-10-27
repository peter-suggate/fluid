import type { FeatureDefinition } from "../../framework/composition";

/**
 * The solve's clock, as the first section of the study rail.
 *
 * The fixed step is the one solve parameter a reader changes while watching
 * the water — to slow a splash down, or to find where a scene goes unstable —
 * and it sat in the transport's overflow menu beside the recorder. The rail is
 * where every setting of the fluid and of its picture lives, so it lives here,
 * a card away from the clock it paces.
 */
export const simulationClockFeature = {
  id: "simulation.clock",
  label: "Simulation clock",
  controls: [
    { id: "step", label: "Step", kind: "number", unit: "ms", update: "live",
      hint: "Fixed simulation step. Rigid bodies and fluid advance on the same step; changes apply live without resetting the clock." },
  ],
  sections: [
    { id: "simulation", surface: "study-rail", slot: "study.simulation", label: "Simulation", order: 0,
      hint: "The solve's clock.", icon: "M12 7v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0" },
  ],
  placements: [{ slot: "study.simulation", control: "step", presentation: "compact" }],
} as const satisfies FeatureDefinition;
