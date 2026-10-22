import type { SvoLightingOptions } from "../../pipeline/svo-render-options";

/** Raster shadows and contact/coarse AO consume no exitant-radiance atlas.
 * The opacity pyramid remains resident for coarse canopy AO. Cone backends
 * retain their complete paired hierarchy, even with indirect light disabled. */
export function svoDerivedRadianceRequired(options: Pick<SvoLightingOptions, "coneTracingMode" | "globalIlluminationEnabled">): boolean {
  return options.coneTracingMode !== "raster-ao" || options.globalIlluminationEnabled === true;
}
