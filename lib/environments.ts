export const environmentIds = [
  "conservatory",
  "courtyard",
  "night-lab",
  "concrete-gallery",
  "bathhouse",
  "research-station",
  "default",
  "garden"
] as const;

export type EnvironmentId = typeof environmentIds[number];

export interface EnvironmentPreset {
  id: EnvironmentId;
  name: string;
  shortName: string;
  description: string;
  /** Three CSS colours used by the compact material swatch in the viewport UI. */
  swatch: readonly [string, string, string];
}

export const environmentPresets: ReadonlyArray<EnvironmentPreset> = [
  {
    id: "default",
    name: "Original fluid studio",
    shortName: "Default",
    description: "The original dark teal studio, neutral grid floor and open environment light.",
    swatch: ["#071514", "#173b37", "#63a99a"]
  },
  {
    id: "conservatory",
    name: "Sunlit conservatory",
    shortName: "Conservatory",
    description: "Limestone, tall garden glazing, warm sun and botanical framing.",
    swatch: ["#183f35", "#d8c99e", "#8cbf7b"]
  },
  {
    id: "courtyard",
    name: "Mediterranean courtyard",
    shortName: "Courtyard",
    description: "Chalky stucco, terracotta tile, cobalt accents and citrus shade.",
    swatch: ["#f1dfbd", "#aa563d", "#245d75"]
  },
  {
    id: "night-lab",
    name: "Night research laboratory",
    shortName: "Research lab",
    description: "The tank on a lab bench at night — warm task light, cool ceiling panels, soft shadows and a city window.",
    swatch: ["#10151c", "#f0dfb8", "#41586a"]
  },
  {
    id: "concrete-gallery",
    name: "Brutalist water gallery",
    shortName: "Gallery",
    description: "Board-formed concrete, a luminous portal and sculptural shadow.",
    swatch: ["#6c716d", "#d79b63", "#202a28"]
  },
  {
    id: "bathhouse",
    name: "Japanese bathhouse",
    shortName: "Bathhouse",
    description: "River stone, cedar battens and softly glowing shoji panels.",
    swatch: ["#47352a", "#d4ba86", "#677b70"]
  },
  {
    id: "research-station",
    name: "Submerged research station",
    shortName: "Research station",
    description: "Pressure ribs, portholes, instrument light and drifting particles.",
    swatch: ["#061724", "#2b6f7b", "#d2a55f"]
  },
  {
    id: "garden",
    name: "Porcelain garden",
    shortName: "Garden",
    description: "A miniature white-clay garden — cloud trees and oversized mushrooms crowd the pond, warm sun on porcelain.",
    swatch: ["#ece9e2", "#b9b6ae", "#8fa0b4"]
  }
];

export const defaultEnvironmentId: EnvironmentId = "default";

export function isEnvironmentId(value: unknown): value is EnvironmentId {
  return typeof value === "string" && environmentIds.includes(value as EnvironmentId);
}

export function getEnvironmentPreset(id: EnvironmentId) {
  return environmentPresets.find((preset) => preset.id === id) ?? environmentPresets[0];
}

export function environmentIndex(id: EnvironmentId) {
  return environmentIds.indexOf(id);
}
