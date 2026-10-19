import type { MethodParamValues } from "../../core/method-contract";
import type { UniformMomentumSampler } from "./uniform-cached-momentum.wgsl";

/** The NB-FLIP near-settled settings (method params regionTiering,
 * warmResolution, particleSeeding, bandExit, bandCycles, rootReserve, rootCoarse, rootTiles, surfaceSplat), fixed for a solver's life (update
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
 * - bandExit: "returns" ends the h band's V-cycles at 0.05 1/s
 *   (UNIFORM_PRESSURE_ARENA_RETURNS_RESIDUAL); "strict" only at its 1e-4 1/s tolerance.
 * - bandCycles: the h band's V-cycles at most (1..8, default 4).
 * - rootReserve: "fused" runs a reserve traversal in one workgroup;
 *   "parallel" retains the ordinary traversal in every safety slot.
 * - rootCoarse: "fused" joins the 16h and 32h levels in one workgroup;
 *   "tiled" selects levels by their allocated row count.
 * - rootTiles: "page" shares root halos across eight octants; "octant"
 *   uses one workgroup per octant. Both perform the same iterations.
 * - surfaceSplat: "fused" selects occupied cells during surface scatter;
 *   "separate" builds a cell list with a full-grid scan first.
 * - traceSubsteps: "gradient" sizes the level-set backtrace's substeps by
 *   the velocity gradient along the step; "travel" takes one a cell of travel.
 *   Particles take one RK3 step either way.
 * - wallTrace: "clip" traces a wall-plane vertex in one RK3 step and takes
 *   its wall contact continuation from the interior point moved by that
 *   trace, clipped into the domain (no second trace); "retrace" substeps it
 *   (traceSubsteps) and traces the interior point again.
 * - surfaceVelocitySampler: "texture32" filters the surface advect's fast
 *   unit velocity samples (advectFine, advectTrace, advectDeferred's fast
 *   twin) in the texture unit, from the same rgba32float halo; "software"
 *   keeps their eight loads. "texture32" needs float32-filterable and falls
 *   back to "software" without it.
 * - redistanceSearch: "shared" lets redistance reuse the crossing search that
 *   retirement ran just before it; "fresh" searches again after retirement
 *   (which can move a hanging zero at a coarse seam).
 * - particleOrder: "scheduled" files samples into their cells in slot order;
 *   "canonical" sorts each cell too (bit-reproducible runs).
 * - momentumSampler: how grid momentum's unit sampler interpolates
 *   (UniformMomentumSampler); "texture32" falls back to "texture16" on a
 *   device without float32-filterable.
 *   "texture32" also filters the particle trace's halo samples in the
 *   texture unit; the other values keep its eight loads.
 */
export interface UniformNearSettled{readonly regionTiering:"error"|"motion";readonly warmResolution:"4h"|"h";readonly particleSeeding:"need"|"activity";readonly sourceSeeding:"need"|"always";readonly bandExit:"returns"|"strict";readonly surfaceSplat:"fused"|"separate";readonly rootTiles:"page"|"octant";readonly rootCoarse:"fused"|"tiled";readonly rootReserve:"fused"|"parallel";readonly bandCycles:number;readonly traceSubsteps:"gradient"|"travel";readonly wallTrace:"clip"|"retrace";readonly surfaceVelocitySampler:"texture32"|"software";readonly redistanceSearch:"shared"|"fresh";readonly particleOrder:"scheduled"|"canonical";readonly momentumSampler:UniformMomentumSampler}

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
 bandCycles:typeof values.bandCycles==="number"&&Number.isInteger(values.bandCycles)?Math.min(8,Math.max(1,values.bandCycles)):4,
 traceSubsteps:values.traceSubsteps==="travel"?"travel":"gradient",
 wallTrace:values.wallTrace==="retrace"?"retrace":"clip",
 surfaceVelocitySampler:values.surfaceVelocitySampler==="software"?"software":"texture32",
 redistanceSearch:values.redistanceSearch==="fresh"?"fresh":"shared",
 particleOrder:values.particleOrder==="canonical"?"canonical":"scheduled",
 momentumSampler:values.momentumSampler==="software"||values.momentumSampler==="texture16"?values.momentumSampler:"texture32",
});
