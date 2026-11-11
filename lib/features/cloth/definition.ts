import type { FeatureDefinition } from "../../framework/composition";

/**
 * The scene's cloth, as a section of the study rail: what a cloth frame costs
 * by stage, how its last step converged, and the Newton-PCG settings that
 * trade the two. Installed while the scene holds cloth, fluid or not.
 */
export const clothFeature = {
  id: "cloth",
  label: "Cloth",
  controls: [
    { id: "cost", label: "Frame", kind: "readout",
      hint: "Measured GPU time of a cloth frame, by stage." },
    { id: "convergence", label: "Convergence", kind: "readout",
      hint: "Newton and PCG iterations, residual and step length of the last sampled step." },
    { id: "settings", label: "Solver", kind: "number",
      hint: "Newton-PCG caps and tolerances; a change carries on from the live pose." },
  ],
  sections: [
    { id: "cloth", surface: "study-rail", slot: "study.cloth", label: "Cloth", order: 10,
      hint: "Cloth frame cost by stage, convergence and solver settings.",
      icon: "M4 6c3-2 5 2 8 0s5-2 8 0v12c-3-2-5 2-8 0s-5-2-8 0z" },
  ],
  placements: [
    { slot: "study.cloth", control: "cost", presentation: "compact" },
    { slot: "study.cloth", control: "convergence", presentation: "compact" },
    { slot: "study.cloth", control: "settings", presentation: "compact" },
  ],
} as const satisfies FeatureDefinition;
