/** Scene presentation modes; sparse voxels remain available for explicit A/B use. */
export const SVO_RENDER_MODES = ["raster", "svo"] as const;

export type SvoRenderMode = typeof SVO_RENDER_MODES[number];

/** Lighting backends for the direct sparse-scene renderer. */
export const SVO_LIGHTING_MODES = ["direct", "cone"] as const;

export type SvoLightingMode = typeof SVO_LIGHTING_MODES[number];

/** User-facing visibility effects layered over the selected lighting backend. */
export type SvoLightingOptions = Readonly<{
  shadowsEnabled: boolean;
  ambientOcclusionEnabled: boolean;
}>;

/** Cone lighting is fail-soft: unavailable caches retain exact direct SVO lighting. */
export const DEFAULT_SVO_LIGHTING_MODE: SvoLightingMode = "cone";

/** The presentation preset aims for the finished image; each effect remains independently switchable. */
export const DEFAULT_SVO_LIGHTING_OPTIONS: SvoLightingOptions = Object.freeze({
  shadowsEnabled: true,
  ambientOcclusionEnabled: true,
});

/**
 * Default to the bounded raster presentation. The SVO dry-scene replacement
 * shares the physics queue and remains opt-in until its worst-case frame cost
 * cannot delay simulation completion or browser interaction.
 */
export const DEFAULT_SVO_RENDER_MODE: SvoRenderMode = "raster";

export function isSvoRenderMode(value: unknown): value is SvoRenderMode {
  return typeof value === "string" && (SVO_RENDER_MODES as readonly string[]).includes(value);
}

export function isSvoLightingMode(value: unknown): value is SvoLightingMode {
  return typeof value === "string" && (SVO_LIGHTING_MODES as readonly string[]).includes(value);
}
