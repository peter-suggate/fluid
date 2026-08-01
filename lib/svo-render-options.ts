/** User-facing visibility effects layered over cone-traced global illumination. */
/**
 * How lighting visibility (shadows, AO, GI) is resolved.
 * - `cones`: hierarchical cone marches plus every stage that feeds them — the
 *   reduced-rate cone prepass, sample fan-out, and persistent world-GI cache.
 * - `exact`: no cone stage runs; shadows and AO use the bounded exact SVO
 *   visibility traversals (sharp reference shadows, costlier per pixel).
 * - `off`: no visibility work at all — direct lighting is unshadowed, AO and
 *   GI are absent. Strictly removes work relative to either other mode.
 */
export type SvoConeTracingMode = "cones" | "exact" | "off";

/**
 * How *primary* visibility is resolved, as distinct from the lighting visibility
 * `SvoConeTracingMode` selects.
 * - `raster`: hardware-rasterize the resident bricks as depth-tested proxies.
 *   Octree leaves partition space, so the depth test alone is an exact
 *   visibility oracle and the image matches `traced` pixel for pixel.
 * - `traced`: the full-screen traversal megakernel every pixel marches for
 *   itself. Kept switchable because it is the reference the raster path is
 *   measured against, and because a device too narrow for four depth-tested
 *   colour planes has to fall back to it.
 */
export type SvoPrimaryTraversalMode = "raster" | "traced";

export type SvoLightingOptions = Readonly<{
  shadowsEnabled: boolean;
  ambientOcclusionEnabled: boolean;
  /** Omitted means `cones`. */
  coneTracingMode?: SvoConeTracingMode;
  /** Omitted means `raster`. Switching it rebuilds the dry-scene pipeline. */
  primaryTraversal?: SvoPrimaryTraversalMode;
}>;

/** The presentation preset aims for the finished image; each effect remains independently switchable. */
export const DEFAULT_SVO_LIGHTING_OPTIONS: SvoLightingOptions = Object.freeze({
  shadowsEnabled: true,
  ambientOcclusionEnabled: true,
  coneTracingMode: "cones",
  primaryTraversal: "raster",
});
