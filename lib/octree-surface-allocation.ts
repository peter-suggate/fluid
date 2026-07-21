export interface OctreeSurfaceStateAllocationPlan {
  readonly presentationOnly: boolean;
  readonly releasePublicationAfterBootstrap: boolean;
  readonly cellCount: number;
  /** Peak bytes of the transient topology bootstrap publication. */
  readonly publicationBytes: number;
  /** Format-compatible public texture retained after bootstrap. */
  readonly persistentPublicationBytes: number;
  /** Legacy scratch, predicted, reversed, and two vec2u jump-flood seed arenas. */
  readonly legacyAuxiliaryBytes: number;
  readonly allocatedBytes: number;
  readonly persistentAllocatedBytes: number;
  readonly denseBaselineBytes: number;
  readonly savedBytes: number;
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
  presentationOnly: boolean,
  releasePublicationAfterBootstrap = false,
  analyticSparseBootstrap = false,
): OctreeSurfaceStateAllocationPlan {
  for (const [axis, value] of dimensions.entries()) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`Surface dimension ${axis} must be a positive safe integer`);
  }
  const cellCount = dimensions[0] * dimensions[1] * dimensions[2];
  if (!Number.isSafeInteger(cellCount)) throw new RangeError("Surface cell count exceeds safe integer range");
  const publicationBytes = cellCount * 4;
  const persistentPublicationBytes = presentationOnly && (releasePublicationAfterBootstrap || analyticSparseBootstrap) ? 4 : publicationBytes;
  // Three r32float textures plus two 8-byte seed records per finest cell.
  const legacyAuxiliaryBytes = cellCount * (3 * 4 + 2 * 8);
  const denseBaselineBytes = publicationBytes + legacyAuxiliaryBytes;
  return {
    presentationOnly,
    releasePublicationAfterBootstrap,
    cellCount,
    publicationBytes,
    persistentPublicationBytes,
    legacyAuxiliaryBytes,
    allocatedBytes: presentationOnly ? (analyticSparseBootstrap ? 4 : publicationBytes) : denseBaselineBytes,
    persistentAllocatedBytes: presentationOnly ? persistentPublicationBytes : denseBaselineBytes,
    denseBaselineBytes,
    savedBytes: presentationOnly ? denseBaselineBytes - persistentPublicationBytes : 0,
  };
}
