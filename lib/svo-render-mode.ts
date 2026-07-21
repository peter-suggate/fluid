/** Scene presentation modes; sparse voxels remain available for explicit A/B use. */
export const SVO_RENDER_MODES = ["raster", "svo"] as const;

export type SvoRenderMode = typeof SVO_RENDER_MODES[number];

/**
 * Default to the bounded raster presentation. The SVO dry-scene replacement
 * shares the physics queue and remains opt-in until its worst-case frame cost
 * cannot delay simulation completion or browser interaction.
 */
export const DEFAULT_SVO_RENDER_MODE: SvoRenderMode = "raster";

export function isSvoRenderMode(value: unknown): value is SvoRenderMode {
  return typeof value === "string" && (SVO_RENDER_MODES as readonly string[]).includes(value);
}
