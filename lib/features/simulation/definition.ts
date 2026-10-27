import type { FeatureDefinition } from "../../framework/composition";

/**
 * The solve, as the first section of the study rail: its clock, what an
 * advance costs, and the rest of its settings folded under that.
 *
 * The cost was a pane of its own over the scene — the advance pipeline — and
 * drew every stage, every chip and every breakdown at once. Here it reads one
 * line per band; a band opens to its stages and a stage to its own settings,
 * so the card is as short as the question being asked.
 */
export const simulationFeature = {
  id: "simulation",
  label: "Simulation",
  controls: [
    { id: "step", label: "Step", kind: "number", unit: "ms", update: "live",
      hint: "Fixed step for fluid and rigid bodies. Applies live." },
    { id: "cost", label: "Advance", kind: "readout",
      hint: "Measured GPU time per advance, by band and stage." },
    { id: "more", label: "More", kind: "number",
      hint: "Grid, solver budgets and detail coverage." },
  ],
  sections: [
    { id: "simulation", surface: "study-rail", slot: "study.simulation", label: "Simulation", order: 0,
      hint: "Step, cost and solver settings.", icon: "M12 7v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0" },
  ],
  placements: [
    { slot: "study.simulation", control: "step", presentation: "compact" },
    { slot: "study.simulation", control: "cost", presentation: "compact" },
    { slot: "study.simulation", control: "more", presentation: "compact" },
  ],
} as const satisfies FeatureDefinition;
