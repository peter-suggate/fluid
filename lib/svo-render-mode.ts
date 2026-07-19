/** Production scene presentation; raster remains the compatibility fallback. */
export const SVO_RENDER_MODES = ["raster", "svo"] as const;

export type SvoRenderMode = typeof SVO_RENDER_MODES[number];

/**
 * Default hybrid presentation: direct SVO dry-scene replacement feeding the
 * existing raster water extraction/interface/optical compositor. `raster`
 * remains an explicit selection and the automatic compatibility fallback.
 */
export const DEFAULT_SVO_RENDER_MODE: SvoRenderMode = "svo";

export function isSvoRenderMode(value: unknown): value is SvoRenderMode {
  return typeof value === "string" && (SVO_RENDER_MODES as readonly string[]).includes(value);
}
