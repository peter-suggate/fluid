export interface OctreeSurfaceStateAllocationPlan {
  readonly releasePublicationAfterBootstrap: boolean;
  readonly cellCount: number;
  /** Peak bytes of the transient topology bootstrap publication. */
  readonly publicationBytes: number;
  /** Format-compatible public texture retained after bootstrap. */
  readonly persistentPublicationBytes: number;
  readonly allocatedBytes: number;
  readonly persistentAllocatedBytes: number;
}

/**
 * Persistent box-sized allocation owned by WebGPUQuadtreeSurfaceState.
 *
 * Small uniform/diagnostic buffers are deliberately excluded: they are
 * constant-size and remain below a kilobyte. This plan captures every term
 * that scales with simulation volume, which is the large/deep-domain concern.
 */
export function planOctreeSurfaceStateAllocation(
  dimensions: readonly [number, number, number],
  _releasePublicationAfterBootstrap = false,
  analyticSparseBootstrap = false,
): OctreeSurfaceStateAllocationPlan {
  for (const [axis, value] of dimensions.entries()) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`Surface dimension ${axis} must be a positive safe integer`);
  }
  const cellCount = dimensions[0] * dimensions[1] * dimensions[2];
  if (!Number.isSafeInteger(cellCount)) throw new RangeError("Surface cell count exceeds safe integer range");
  const publicationBytes = cellCount * 4;
  // Non-analytic bootstrap phi remains bound by recurring WebGPU groups even
  // after Section 5's fine SPGrid and background octree become authoritative.
  // It may only be excluded when analytic bootstrap allocated one texel from
  // the outset; treating an unused binding as dead is invalid in Dawn.
  const persistentPublicationBytes = analyticSparseBootstrap ? 4 : publicationBytes;
  return {
    releasePublicationAfterBootstrap: analyticSparseBootstrap,
    cellCount,
    publicationBytes,
    persistentPublicationBytes,
    allocatedBytes: analyticSparseBootstrap ? 4 : publicationBytes,
    persistentAllocatedBytes: persistentPublicationBytes,
  };
}
