import type { MethodParamValues } from "../../core/method-contract";

/** The NB-FLIP near-settled settings (method params regionTiering,
 * warmResolution, particleSeeding, bandExit, bandSmoothing, rootReserve, rootCoarse, rootTiles, surfaceSplat), fixed for a solver's life (update
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
 *   (UniformPressureBandFields.returns); "strict" only at its 1e-4 1/s tolerance.
 * - bandSmoothing: "fused" shares small-band 2h visits within one workgroup;
 *   "separate" dispatches each colour. Both perform the same iterations.
 * - rootReserve: "fused" runs a reserve traversal in one workgroup;
 *   "parallel" retains the ordinary traversal in every safety slot.
 * - rootCoarse: "fused" joins the 16h and 32h levels in one workgroup;
 *   "tiled" selects levels by their allocated row count.
 * - rootTiles: "page" shares root halos across eight octants; "octant"
 *   uses one workgroup per octant. Both perform the same iterations.
 * - surfaceSplat: "fused" selects occupied cells during surface scatter;
 *   "separate" builds a cell list with a full-grid scan first.
 * - traceSubsteps: "gradient" sizes particle and level-set trace substeps by
 *   the velocity gradient along the step; "travel" takes one a cell of travel.
 * - redistanceSearch: "shared" lets redistance reuse the crossing search that
 *   retirement ran just before it; "fresh" searches again after retirement
 *   (which can move a hanging zero at a coarse seam).
 * - particleDeposit: "tiled" scatters each occupied tile's particle momentum
 *   into shared fixed-point sums flushed once; "gathered" gathers each fine
 *   cell's particle neighbourhood (the same result to fixed-point rounding).
 */
export interface UniformNearSettled{readonly regionTiering:"error"|"motion";readonly warmResolution:"4h"|"h";readonly particleSeeding:"need"|"activity";readonly sourceSeeding:"need"|"always";readonly bandExit:"returns"|"strict";readonly surfaceSplat:"fused"|"separate";readonly rootTiles:"page"|"octant";readonly rootCoarse:"fused"|"tiled";readonly rootReserve:"fused"|"parallel";readonly bandSmoothing:"fused"|"separate";readonly traceSubsteps:"gradient"|"travel";readonly redistanceSearch:"shared"|"fresh";readonly particleDeposit:"tiled"|"gathered"}

export const uniformNearSettled=(values:MethodParamValues={}):UniformNearSettled=>({
 regionTiering:values.regionTiering==="motion"?"motion":"error",
 warmResolution:values.warmResolution==="4h"?"4h":"h",
 particleSeeding:values.particleSeeding==="activity"?"activity":"need",
 sourceSeeding:values.sourceSeeding==="always"?"always":"need",
 bandExit:values.bandExit==="strict"?"strict":"returns",
 surfaceSplat:values.surfaceSplat==="separate"?"separate":"fused",
 rootTiles:values.rootTiles==="octant"?"octant":"page",
 rootCoarse:values.rootCoarse==="tiled"?"tiled":"fused",
 rootReserve:values.rootReserve==="parallel"?"parallel":"fused",
 bandSmoothing:values.bandSmoothing==="separate"?"separate":"fused",
 traceSubsteps:values.traceSubsteps==="travel"?"travel":"gradient",
 redistanceSearch:values.redistanceSearch==="fresh"?"fresh":"shared",
 particleDeposit:values.particleDeposit==="gathered"?"gathered":"tiled",
});
