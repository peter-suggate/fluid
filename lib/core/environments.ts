export const environmentIds = [
  "conservatory",
  "courtyard",
  "night-lab",
  "concrete-gallery",
  "bathhouse",
  "research-station",
  "default",
  "garden",
  // Appended: environment indices are packed into GPU records and glass pane
  // ids, so an id keeps its position for as long as it exists.
  "stage"
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

/**
 * Every set is built in near-neutral white so that shading, shadow and ambient
 * occlusion carry the form, and the water is the only thing in frame with
 * colour and motion of its own. Each swatch is therefore a value ramp — dark,
 * mid, light — plus whatever tint that environment's light sources bring.
 */
export const environmentPresets: ReadonlyArray<EnvironmentPreset> = [
  {
    id: "default",
    name: "White room",
    shortName: "White room",
    description: "A plain white enclosure with one overhead light and no set dressing.",
    swatch: ["#d9d9d9", "#eeeeee", "#ffffff"]
  },
  {
    id: "stage",
    name: "Spotlit stage",
    shortName: "Stage",
    description: "A dark stage and one lamp hung over it: no walls, no set dressing, and a pool of light the size of whatever stands in it.",
    swatch: ["#1a1a1a", "#4a4744", "#f2ece2"]
  },
  {
    id: "conservatory",
    name: "Sunlit conservatory",
    shortName: "Conservatory",
    description: "A white glasshouse — tall mullions cut hard bars of sun across the tank while the jet climbs the fill line.",
    swatch: ["#6e7370", "#c9cbc6", "#f6f2e8"]
  },
  {
    id: "courtyard",
    name: "Mediterranean courtyard",
    shortName: "Courtyard",
    description: "Chalky white stucco and a colonnade at noon: hard shadow, bright bounce, and a rill running to the tank.",
    swatch: ["#7a7469", "#cfc8ba", "#f7f2e6"]
  },
  {
    id: "night-lab",
    name: "Night research laboratory",
    shortName: "Research lab",
    description: "The tank as specimen — warm task light against cool ceiling panels, a camera trained on the glass, and a city window behind.",
    swatch: ["#1b1f24", "#8f959b", "#f0dfb8"]
  },
  {
    id: "concrete-gallery",
    name: "Brutalist water gallery",
    shortName: "Gallery",
    description: "White board-formed concrete and a luminous slot: the room is a plinth, and the dam break is the exhibit.",
    swatch: ["#20242a", "#8b8f8c", "#e8c9a4"]
  },
  {
    id: "bathhouse",
    name: "Japanese bathhouse",
    shortName: "Bathhouse",
    description: "Bleached cedar battens stripe the lantern light across still water. The calmest set in the app.",
    swatch: ["#4c4a45", "#b6b1a6", "#f4e6cc"]
  },
  {
    id: "research-station",
    name: "Submerged research station",
    shortName: "Research station",
    description: "A white pressure hull — ribs, bolted flanges and a depth gauge climbing the wall, all built to hold back a great deal of water.",
    swatch: ["#2d3438", "#9aa3a6", "#cfe6ea"]
  },
  {
    id: "garden",
    name: "Porcelain garden",
    shortName: "Garden",
    description: "A miniature white-clay garden — cloud trees, oversized mushrooms and reeds crowd the pond, warm sun on porcelain.",
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
