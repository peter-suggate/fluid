/** Elide contact queries only for the move's proven empty swept hull.
 * Keep the walk's original half-cell additions: replacing them with its
 * endpoint changes rounding. Other entry points and flat jobs fall back. */
export const narrowBandClearWalkWGSL=/* wgsl */`
fn nbClearWalk()->bool{
 // Procedural contact uses its own field/gradient rule; terrain has its
 // own skin. Neither is certified by this empty voxel/body hull proof.
 if(!nbWinOn||umSolidHasTerrain()||umBoundaryBase()!=0u){return false;}
 // Match nbHullUncut's existing guard around the swept particle sphere,
 // including the walk's rounded increments, in the body rejection too.
 let low=nbHullLo-vec3f(0.001);let high=nbHullHi+vec3f(0.001);
 let middle=0.5*(low+high);let radius=length(0.5*(high-low)*params.hDt.xyz);
 return !umAnyBodyNear(umSolidTraceWorld(middle),radius)&&nbHullUncut();
}
`;
