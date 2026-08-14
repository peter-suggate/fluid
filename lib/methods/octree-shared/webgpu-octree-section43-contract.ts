import type { PassBroker } from "../../core/webgpu-pass-broker";

export const OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS = 8;
export const OCTREE_SECTION43_BOUNDARY_BAND_LAYERS = 3;
export const OCTREE_FIRST_ORDER_CHEBYSHEV_DEGREES = Object.freeze([2, 4] as const);
export const OCTREE_FIRST_ORDER_CHEBYSHEV_LOWER_FRACTION = 1 / 30;
/**
 * The first-order Setaluri V-cycle contract consumed by the production
 * Section 4.3 fixed-schedule preconditioner. It is deliberately solver-agnostic: the
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
  /**
   * Optional split encoding for callers that already own a convergence-gate
   * publication pass. Both methods must be supplied together. The gate writes
   * only indirect records; the body consumes them after the caller's fence.
   */
  encodeCorrectionGate?(pass: GPUComputePassEncoder, input: {
    readonly rhs: GPUBuffer;
    readonly correction: GPUBuffer;
    readonly solverControl: GPUBuffer;
    readonly rowCount: GPUBuffer;
  }): void;
  encodeCorrectionBody?(broker: PassBroker, input: {
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
