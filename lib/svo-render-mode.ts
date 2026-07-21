/** Scene presentation modes; raster remains available for explicit A/B use. */
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

/** WebGPU presents the unified sparse scene by default; raster is an explicit fallback/A-B choice. */
export const DEFAULT_SVO_RENDER_MODE: SvoRenderMode = "svo";

export function isSvoRenderMode(value: unknown): value is SvoRenderMode {
  return typeof value === "string" && (SVO_RENDER_MODES as readonly string[]).includes(value);
}

export function isSvoLightingMode(value: unknown): value is SvoLightingMode {
  return typeof value === "string" && (SVO_LIGHTING_MODES as readonly string[]).includes(value);
}
