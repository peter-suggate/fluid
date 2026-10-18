import type { MethodParamValues } from "../../core/method-contract";

/** The NB-FLIP near-settled settings (method params regionTiering,
 * warmResolution, particleSeeding), fixed for a solver's life (update
 * "solver": pipeline variants are chosen at attach). Each default is the
 * resolution-proportional behaviour; the other option is the motion-driven
 * treatment before it, kept as a maintained path.
 * - regionTiering: "error" tiers a 16h region by the resolution its motion
 *   needs (UniformRegionHeatSources.tiering); "motion" by any motion.
 * - warmResolution: "4h" computes WARM regions at 4h; "h" at h.
 * - particleSeeding: "need" seeds particles only where the grid under-resolves
 *   the flow; "activity" wherever the activity heat spreads. */
export interface UniformNearSettled{readonly regionTiering:"error"|"motion";readonly warmResolution:"4h"|"h";readonly particleSeeding:"need"|"activity"}

export const uniformNearSettled=(values:MethodParamValues={}):UniformNearSettled=>({
 regionTiering:values.regionTiering==="motion"?"motion":"error",
 warmResolution:values.warmResolution==="h"?"h":"4h",
 particleSeeding:values.particleSeeding==="activity"?"activity":"need",
});
