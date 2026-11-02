/** A host change between mixed frames, as UniformMixedFrame.invalidate takes it.
 * solids: a live voxel edit rewrote the solid mask. fluid: liquid injected or
 * a field uploaded. values: runtime parameters. scene: scene uniforms or the
 * topology fence. bodies: rigid body poses (every presented frame of an
 * interaction scene: it reaches only what a body frame's head does not
 * already rebuild). layout: a host relayout. capacity: fine or band storage
 * reallocated. relayout: a GPU relayout attached or detached. census: a
 * census whose receipts do not count yet (its first build, or an uncounted
 * attach): the lagged work evidence of the previous owner is forgotten.
 * coarsePhi: the 4h vertex phi base republished. */
export type UniformMixedHostChange="solids"|"fluid"|"values"|"scene"|"bodies"|"layout"|"capacity"|"relayout"|"census"|"coarsePhi";

/** A host-side cache the frame keeps over GPU state, and what reaching it does.
 * cold: the cold frame's evidence (trigger). workControl: surface work
 * control's retained plans and revisions. dynamics: region dynamics. heat:
 * region heat (every region HOT). heldAuthority / splitAuthority: the
 * surface-only and all-4h pressure authorities' retained regions.
 * extension: the census extension's reuse. geometry: the simulation target
 * and centre phi's currency. solidRecord: the all-4h solid record.
 * solidWidths: the cut widths. workEpoch, bandWork, bandCycles, ownershipWork,
 * residency, remapWork: the lagged work evidence launch widths size from. */
export type UniformMixedCache="cold"|"workControl"|"dynamics"|"heat"|"heldAuthority"|"splitAuthority"|"extension"|"geometry"|"solidRecord"|"solidWidths"
 |"workEpoch"|"bandWork"|"bandCycles"|"ownershipWork"|"residency"|"remapWork";

/** What a write of phi, volume or parameters stales: everything retained over the fluid. */
const FLUID:readonly UniformMixedCache[]=["cold","heat","heldAuthority","splitAuthority","extension","geometry"];
/** The one map from host changes to the caches they stale, in reach order. */
export const UNIFORM_MIXED_REACHES:{readonly [change in UniformMixedHostChange]:readonly UniformMixedCache[]}={
 solids:["workControl","dynamics","solidRecord","solidWidths",...FLUID],
 fluid:FLUID,
 values:FLUID,
 scene:FLUID,
 // A body frame's head rebuilds the record, heat and held authority for the swept tiles itself (noteBodies).
 bodies:["cold","splitAuthority","extension","geometry"],
 layout:["workControl","dynamics","solidWidths",...FLUID],
 capacity:["cold"],
 relayout:["cold"],
 // workEpoch first: the others size from receipts after it.
 census:["workEpoch","bandWork","bandCycles","ownershipWork","residency","remapWork"],
 coarsePhi:["workControl","dynamics"],
};
