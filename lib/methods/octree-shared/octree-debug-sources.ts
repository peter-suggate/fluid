import type { GPUSolverInstance } from "../../core/method-contract";

/**
 * The octree lane's QA publications, as the lane itself sees them.
 *
 * None of this is part of the renderer's solver contract: these are producer
 * headers, arenas and forensic scratch buffers whose meaning is the octree
 * method's alone. The contract carries them as an opaque bag so that core code
 * never names a lane; this module is the other half of that seam, and the only
 * place the bag's real shape is written down.
 *
 * Every member is optional because a publication exists only once the
 * subsystem that owns it has been constructed. A solve that never builds the
 * Losasso backend, or one inspected before its first accepted step, simply has
 * fewer entries — absent means "no such producer here", never "the producer
 * reported nothing".
 *
 * Declared as a type alias rather than an interface so it widens to the bag's
 * `Record<string, unknown>` without an assertion at the publishing end.
 */
export type OctreeDebugSources = {
  /** QA-only active compact pressure potential, indexed by power-leaf row. */
  readonly powerPressureBuffer?: GPUBuffer;
  /** QA-only compact leaf headers; 48 bytes per pressure row. */
  readonly powerLeafHeaders?: GPUBuffer;
  /** Diagnostic-only; never participates in authority selection. */
  readonly globalFineTransportControl?: GPUBuffer;
  readonly globalFineRedistanceControl?: GPUBuffer;
  readonly globalFineVolumeControl?: GPUBuffer;
  readonly structuredProjectionEnergyStats?: GPUBuffer;
  /**
   * QA-only native Losasso axis-face authority for Dawn field reconstruction.
   *
   * The first four members reconstruct the accepted extended field; the `wet`
   * members expose the same faces at each stage of the step (advected,
   * predicted, projected, extended) so a harness can attribute a velocity
   * defect to the producer that introduced it.
   */
  readonly losassoVelocityDebug?: {
    readonly control: GPUBuffer;
    readonly faceGeometry: GPUBuffer;
    readonly projectedVelocity: GPUBuffer;
    readonly extendedVelocity: GPUBuffer;
    /**
     * The band's own metric row: which authority published each face, at which
     * dilation layer, and the |phi| that decided it. A support asymmetry has to
     * be read against this — the value receipt cannot tell an absent face from
     * a face published with a different ring stamp. `faceCapacity` comes along
     * so a dropped face stays separable from a face never proposed.
     */
    readonly faceMetrics?: GPUBuffer;
    readonly faceCapacity?: number;
    readonly wetControl: GPUBuffer;
    readonly wetFaceGeometry: GPUBuffer;
    readonly wetAdvectedVelocity: GPUBuffer;
    readonly wetPredictedVelocity: GPUBuffer;
    readonly wetProjectedVelocity: GPUBuffer;
    readonly wetExtendedVelocity: GPUBuffer;
    readonly dimensions: readonly [number, number, number];
    readonly maximumLeafSize: number;
  };
  /**
   * QA-only accepted Losasso row operator, with no Power structured ABI.
   *
   * `control` word 1 counts the accepted rows and word 2 the accepted faces,
   * which is what sizes every other member: the row-indexed unknowns
   * (`rightHandSide`, `diagonal`, `leafHeaders`, `rowPhi`) and the face-indexed
   * geometry (`faces`, `faceGeometry`, `ghostDistances`).
   */
  readonly losassoPressureDebug?: {
    readonly control: GPUBuffer;
    readonly rightHandSide: GPUBuffer;
    readonly diagonal: GPUBuffer;
    readonly faces: GPUBuffer;
    readonly faceGeometry: GPUBuffer;
    readonly leafHeaders: GPUBuffer;
    readonly rowPhi: GPUBuffer;
    readonly ghostDistances: GPUBuffer;
  };
  /** QA-only compact topology-delta streams for rejection forensics. */
  readonly globalFinePageDeltaDebug?: {
    readonly buffer: GPUBuffer;
    readonly params: GPUBuffer;
    readonly sparseCandidates: GPUBuffer;
    readonly sparseCandidateCapacity: number;
    readonly pageCapacity: number;
    readonly changedKeysOffsetWords: number;
    readonly dirtyPagesOffsetWords: number;
    readonly supportPagesOffsetWords: number;
    readonly promotionCountsOffsetWords: number;
  };
  /** QA-only sparse owner-page arena readback for topology forensics. */
  readonly ownerLatticeDebug?: {
    buffer: GPUBuffer;
    maximumLeafSize: number;
    dimensions: readonly [number, number, number];
  };
  readonly globalFineCoarseLevelSetControl?: GPUBuffer;
  readonly globalFineRestrictionControl?: GPUBuffer;
  /** QA-only first-failure receipt for Section-5 air-support publication. */
  readonly airSupportScratch?: GPUBuffer;
};

/**
 * Read the octree lane's publications off any solver.
 *
 * Deliberately total. A solver from another method carries no bag at all, and
 * one from this lane carries only what its subsystems have built, so both
 * decode to "ask and get `undefined`" — the contract every QA readback here
 * was already written against when these were thirteen optional members of the
 * renderer's solver interface. Validating instead would turn a diagnostic that
 * is legitimately absent into a harness failure.
 */
export function octreeDebugSources(solver: GPUSolverInstance): Partial<OctreeDebugSources> {
  // The contract deliberately forgets what the bag holds so core code cannot
  // name a lane. Reattaching the lane's own shape is therefore an assertion by
  // construction, and this is the single place it is allowed to happen: the
  // producer on the other side is the octree solver's `debug` accessor, which
  // is typed against this same declaration.
  return (solver.debug as Partial<OctreeDebugSources> | undefined) ?? {};
}
