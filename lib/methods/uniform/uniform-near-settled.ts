import type { MethodParamValues } from "../../core/method-contract";

/** The NB-FLIP near-settled settings (method params regionTiering,
 * warmResolution, particleSeeding, bandExit), fixed for a solver's life (update
 * "solver": pipeline variants are chosen at attach). Defaults use error-driven update activity and independently selected
 * spatial resolution; comparison options remain available.
 * - regionTiering: "error" bounds omitted transport, momentum and pressure
 *   response; "motion" wakes on any motion.
 * - warmResolution: "h" preserves spatially selected detail (default); "4h"
 *   forcibly coarsens WARM pressure and solid geometry.
 * - particleSeeding: "need" seeds particles only where the grid under-resolves
 *   the flow; "activity" wherever the activity heat spreads.
 * - sourceSeeding: "need" heats a source's tiles only where the census
 *   triggers, like any other liquid; "always" keeps every source tile hot.
 * - bandExit: "returns" ends the h band's V-cycles at diminishing returns
 *   (UniformPressureBandFields.returns); "strict" only at its 1e-4 1/s tolerance. */
export interface UniformNearSettled{readonly regionTiering:"error"|"motion";readonly warmResolution:"4h"|"h";readonly particleSeeding:"need"|"activity";readonly sourceSeeding:"need"|"always";readonly bandExit:"returns"|"strict"}

export const uniformNearSettled=(values:MethodParamValues={}):UniformNearSettled=>({
 regionTiering:values.regionTiering==="motion"?"motion":"error",
 warmResolution:values.warmResolution==="4h"?"4h":"h",
 particleSeeding:values.particleSeeding==="activity"?"activity":"need",
 sourceSeeding:values.sourceSeeding==="always"?"always":"need",
 bandExit:values.bandExit==="strict"?"strict":"returns",
});
