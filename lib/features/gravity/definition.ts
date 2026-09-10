import { GRAVITY_DIRECTIONS } from "./state";
import type { FeatureDefinition } from "../../framework/composition";

export const gravityFeature = {
  id: "scene.gravity",
  label: "Gravity",
  provides: ["physics.acceleration"],
  controls: [
    { id: "enabled", label: "Gravity", kind: "toggle", setting: "fluid.gravity_m_s2", update: "reset",
      hint: "Turn gravity on or off. Scenes starting without gravity use Earth gravity downward." },
    { id: "y", label: "Gravity Y", kind: "number", setting: "fluid.gravity_m_s2.y", unit: "m/s²", step: 0.1, min: -20, max: 0, update: "reset" },
    { id: "direction", label: "Direction", kind: "choice", setting: "fluid.gravity_m_s2", update: "live",
      options: GRAVITY_DIRECTIONS.map(({ id, label }) => ({ value: id, label })) },
  ],
  placements: [{ slot: "scene.physics", control: "enabled", presentation: "compact", priority: "high" },
    { slot: "fluid.material", control: "y", presentation: "expanded" }],
} as const satisfies FeatureDefinition;
