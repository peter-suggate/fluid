/** The move's existing 4³-tile window shares body broad-phase results.
 * Bits 8..19 of nbWinFlags hold the tile's conservative body candidates;
 * its existing low flags and workgroup synchronization are unchanged. */
export const narrowBandBodyWindowWGSL=/* wgsl */`
fn nbBodyTileMask(t:vec3i)->u32{
 // A tile extends two cells from its centre on each axis. Three cells
 // cover every point and the largest particle's half-cell contact skin,
 // including anisotropic cells. umBodyNear always includes interiors.
 return umBodyCandidates(umSolidTraceWorld(vec3f(4*t)+2.0),length(3.0*params.hDt.xyz));
}
fn nbMoveBodyMask(q:vec3f)->u32{
 let w=nbWinIndex(vec3i(floor(q))>>vec3u(2u));
 if(w<64u){return nbWinFlags[w]>>8u;}
 return (1u<<umBodyCount())-1u;
}
fn nbMoveBodyAt(q:vec3f)->i32{return umBodyAtCandidates(umSolidTraceWorld(q),nbMoveBodyMask(q));}
`;
