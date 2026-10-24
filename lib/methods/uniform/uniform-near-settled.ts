import type { MethodParamValues } from "../../core/method-contract";
import type { UniformMomentumSampler } from "./uniform-cached-momentum.wgsl";

/** The NB-FLIP near-settled settings (method params regionTiering,
 * warmResolution, particleSeeding, bandExit, bandCycles, surfaceImpulse, rootReserve, rootCoarse, rootTiles, surfaceSplat), fixed for a solver's life (update
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
 * - bandCycles: the h band's V-cycles at most (1..8, default 3).
 * - pressureTail: "auto" solves the band's coarsest correction over its 32h
 *   rows (a row a page) up to 4096 pages and over the super-block tail's
 *   128h rows above; "superblocks" always takes the super-block tail.
 * - surfaceImpulse: "energy" bounds the speed liquid leaves a free surface
 *   by its pressure's work (UniformPressureArenaSchedule); "full" applies the
 *   step's whole ghost-fluid impulse.
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
 * - surfaceTraceSubsteps: "off" traces a surface vertex back in one RK3
 *   step of any length (split only where a span would cross more than one
 *   page face an axis, 32 cells of travel); "legacy" sizes substeps by
 *   traceSubsteps.
 * - surfaceRedistanceReach: tiles from a crossing tile the redistance
 *   writes exact distances to (1, 2 or 3); past it a vertex keeps its sign
 *   at six cells. Every consumer reads the reach-1 ring.
 * - redistanceSearch: "shared" lets redistance reuse the crossing search that
 *   retirement ran just before it; "fresh" searches again after retirement
 *   (which can move a hanging zero at a coarse seam).
 * - crossingGuard: "previous" takes couple's bulk depth guard (bulkDepth)
 *   from the last step's arena search (its reach bits and nearest-crossing
 *   bank), shortened by the spread reach of this step's travel, and runs no
 *   crossing search on the advected phi; "search" searches the advected phi.
 * - flipDelta: "swept" records each sample's FLIP delta at the tail from the
 *   paired extension's final slots (the current minus the old payload where
 *   the sweeps reached), with no old-field publication or snapshot;
 *   "extended" publishes the old field and samples it, and G2P subtracts
 *   that sample from the next step's PIC.
 * - particleBandDepth: cells below the surface the particle band reaches
 *   (seeding fills it, the move retires samples past it); 2 to 4.
 * - particleDensity: "uniform" seeds eight sites a cell through the band;
 *   "graded" four in cells deeper than 1.5 cells, caps crowding there at
 *   six, and lets faces below a cell expect half the mass.
 * - solidContact: "liquid" holds a static solid's cut tiles, and their
 *   neighbours, at h wherever liquid can meet them; "surface" only where the
 *   surface census already holds the cut tile at h (submerged cut tiles run
 *   as cut 4h owners), and their ring only where it is wet.
 * - momentumSampler: how grid momentum's unit sampler interpolates
 *   (UniformMomentumSampler); "texture32" falls back to "texture16" on a
 *   device without float32-filterable.
 */
export interface UniformNearSettled{readonly regionTiering:"error"|"motion";readonly warmResolution:"4h"|"h";readonly particleSeeding:"need"|"activity";readonly sourceSeeding:"need"|"always";readonly bandExit:"returns"|"strict";readonly surfaceSplat:"fused"|"separate";readonly rootTiles:"page"|"octant";readonly rootCoarse:"fused"|"tiled";readonly rootReserve:"fused"|"parallel";readonly bandCycles:number;readonly pressureTail:"auto"|"superblocks";readonly surfaceImpulse:"energy"|"full";readonly traceSubsteps:"gradient"|"travel";readonly wallTrace:"clip"|"retrace";readonly surfaceVelocitySampler:"texture32"|"software";readonly surfaceRedistanceReach:1|2|3;readonly surfaceTraceSubsteps:"off"|"legacy";readonly redistanceSearch:"shared"|"fresh";readonly crossingGuard:"previous"|"search";readonly flipDelta:"swept"|"extended";readonly momentumSampler:UniformMomentumSampler;readonly solidContact:"liquid"|"surface";readonly particleBandDepth:number;readonly particleDensity:"uniform"|"graded"}

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
 bandCycles:typeof values.bandCycles==="number"&&Number.isInteger(values.bandCycles)?Math.min(8,Math.max(1,values.bandCycles)):3,
 pressureTail:values.pressureTail==="superblocks"?"superblocks":"auto",
 surfaceImpulse:values.surfaceImpulse==="full"?"full":"energy",
 traceSubsteps:values.traceSubsteps==="travel"?"travel":"gradient",
 wallTrace:values.wallTrace==="retrace"?"retrace":"clip",
 surfaceVelocitySampler:values.surfaceVelocitySampler==="software"?"software":"texture32",
 surfaceRedistanceReach:values.surfaceRedistanceReach===1||values.surfaceRedistanceReach===2||values.surfaceRedistanceReach===3?values.surfaceRedistanceReach:3,
 surfaceTraceSubsteps:values.surfaceTraceSubsteps==="legacy"?"legacy":"off",
 redistanceSearch:values.redistanceSearch==="fresh"?"fresh":"shared",
 crossingGuard:values.crossingGuard==="search"?"search":"previous",
 flipDelta:values.flipDelta==="extended"?"extended":"swept",
 momentumSampler:values.momentumSampler==="software"||values.momentumSampler==="texture16"?values.momentumSampler:"texture32",
 solidContact:values.solidContact==="liquid"?"liquid":"surface",
 particleBandDepth:typeof values.particleBandDepth==="number"&&Number.isFinite(values.particleBandDepth)?Math.min(4,Math.max(2,values.particleBandDepth)):3,
 particleDensity:values.particleDensity==="uniform"?"uniform":"graded",
});
