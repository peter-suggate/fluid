/**
 * Proven extents of the generated octree discretization catalog.
 *
 * The shared octree engine bakes these into WGSL loop ceilings and into ABI
 * equality checks, so they must be literals available when a shader string is
 * built -- there is no point at which a backend could hand them over first.
 * They live here rather than in the backend that generates the catalog because
 * a shared engine may not import a lane.
 *
 * The generated manifest remains the authority. `lib/methods/power/
 * octree-power-catalog-extent-agreement.ts` assigns this object to the
 * manifest's own literal types, so regenerating the catalog with different
 * maxima fails `tsc` there instead of silently under-bounding a loop here.
 */
export const OCTREE_CATALOG_EXTENTS = Object.freeze({
  /** Global byte-selector table extent; the direct selector-indexed row width. */
  tetrahedronVertexCount: 75,
  /** Largest tetrahedron count over every catalog entry. A header claiming
   * more than this is counted as invalid rather than clamped away, so the
   * audit overlay fails closed on a catalog it was not built against. */
  maximumTetrahedra: 68,
  /** Largest face count over every catalog entry, and therefore the fixed
   * per-row face-slot ceiling every catalog face walk unrolls to. */
  maximumFaceIncidence: 30,
} as const);
