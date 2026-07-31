/** User-facing visibility effects layered over cone-traced global illumination. */
export type SvoLightingOptions = Readonly<{
  shadowsEnabled: boolean;
  ambientOcclusionEnabled: boolean;
}>;

/** The presentation preset aims for the finished image; each effect remains independently switchable. */
export const DEFAULT_SVO_LIGHTING_OPTIONS: SvoLightingOptions = Object.freeze({
  shadowsEnabled: true,
  ambientOcclusionEnabled: true,
});
