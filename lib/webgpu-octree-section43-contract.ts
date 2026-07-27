import type { PassBroker } from "./webgpu-pass-broker";

export const OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS = 8;
export const OCTREE_SECTION43_BOUNDARY_BAND_LAYERS = 3;
export const OCTREE_FIRST_ORDER_CHEBYSHEV_DEGREES = Object.freeze([2, 4] as const);
export const OCTREE_FIRST_ORDER_CHEBYSHEV_LOWER_FRACTION = 1 / 30;
/**
 * Row-capacity ceiling for the single-dispatch persistent executor
 * (`WebGPUOctreePersistentMGPCG`). The kernel is row-striped —
 * `for (row = lane; row < rowCount; row += 256)` — so capacity is bounded by
 * how much work one workgroup (one GPU core) can absorb before the
 * row-parallel hierarchical path saturates the device, not by the lane count.
 * Above this, the production row-parallel pipelined solver is selected.
 */
export const OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY = 8_192;
/** x, r, z, d, A*d, four hybrid f32 fields, and two band u32 fields. */
export const OCTREE_PERSISTENT_MGPCG_STATE_CHANNELS = 11;

export interface OctreePersistentMGPCGExecutor {
  readonly maximumRowCapacity: number;
  readonly encodedDispatchCount: 1;
  readonly dispatchShape: readonly [1, 1, 1];
  readonly invariantProof: Readonly<{
    ghostRows: "spgrid-identical";
    transfers: "validated-adjoint-pair";
    invalidRows: "uniform-fail-closed-before-arithmetic";
  }>;
  /**
   * The seed/publication pair is the entire per-solve input. Geometry,
   * topology, coefficients, the RHS, the accepted authority and the shared
   * solve-control buffer are all constructor-time bindings, and the row count
   * is read GPU-side from the accepted structured control — the executor
   * never shapes its encode from a readback.
   *
   * `encodeInnerSetup` runs before the executor stages any authority word, so
   * a caller whose SPGrid topology setup/commit has not yet been encoded this
   * step can supply it.
   */
  encodeSolve(broker: PassBroker, input: {
    readonly pressureSeed: GPUBuffer;
    readonly pressureOut: GPUBuffer;
  }, encodeInnerSetup?: (broker: PassBroker) => void): void;
}

/**
 * The first-order Setaluri V-cycle contract consumed by the production
 * Section 4.3 hybrid preconditioner. It is deliberately solver-agnostic: the
 * retired staged MGPCG implementation is not part of this module.
 */
export interface OctreeFirstOrderSPDVCycle {
  readonly operatorOrder: 1;
  readonly isSymmetricPositiveDefinite: true;
  /**
   * Post-convergence iterations must dispatch zero workgroups rather than be
   * skipped at encode time. The preconditioner enforces this on its inner
   * cycle; declaring it here is what lets that check read the member directly
   * instead of casting the contract away.
   */
  readonly convergenceTail: "gpu-zero-indirect";
  readonly allocatedBytes: number;
  readonly encodedCorrectionDispatchCount: number;
  readonly encodedSetupDispatchCount?: number;
  readonly encodedPassTransitionCount?: number;
  readonly persistentMGPCG?: OctreePersistentMGPCGExecutor;
  /**
   * Immutable smoothing proof published by a first-order hierarchy.  The
   * Chebyshev interval is derived from the diagonally scaled rediscretized
   * operator. No alternate smoother is executable.
   */
  readonly smootherContract?: Readonly<{
    kind: "chebyshev";
    degree: 2 | 4;
    spectralBounds: "transactional-scaled-gershgorin";
    lowerFraction: typeof OCTREE_FIRST_ORDER_CHEBYSHEV_LOWER_FRACTION;
  }>;
  encodeSetup(broker: PassBroker, input: {
    readonly solverControl: GPUBuffer;
    readonly rowCount: GPUBuffer;
  }): void;
  encodeCorrection(broker: PassBroker, input: {
    readonly rhs: GPUBuffer;
    readonly correction: GPUBuffer;
    readonly solverControl: GPUBuffer;
    readonly rowCount: GPUBuffer;
  }): void;
}

export function normalizeOctreeSection43BoundarySmoothing(
  value: number | undefined,
): number {
  const requested = value ?? OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS;
  return Math.max(2, Math.min(16, Math.round(requested / 2) * 2));
}
