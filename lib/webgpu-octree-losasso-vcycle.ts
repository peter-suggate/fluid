import type { PassBroker } from "./webgpu-pass-broker";
import type { OctreeFirstOrderSPDVCycle } from "./webgpu-octree-section43-contract";

/** Symmetric plain-cycle shape; the wrapped hierarchy retains its proven smoother. */
export const OCTREE_LOSASSO_VCYCLE_SCHEDULE = Object.freeze({
  preRelaxations: 2,
  postRelaxations: 2,
  coarseSolve: "fixed" as const,
} as const);

export interface OctreeLosassoVCycleLevelSource {
  readonly rowCapacity: number;
  readonly control: GPUBuffer;
  readonly rowFaceOffsets: GPUBuffer;
  readonly rowFaces: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly rowDispatch: GPUBuffer;
}

export interface OctreeLosassoVCycleTransferSource {
  /** Fine row -> one parent coarse row. */
  readonly fineParents: GPUBuffer;
  /** Parent -> ascending fine-row child range. */
  readonly childOffsets: GPUBuffer;
  /** Fine row IDs grouped by parent and sorted ascending within each range. */
  readonly childList: GPUBuffer;
  readonly fineRowDispatch: GPUBuffer;
}

export interface OctreeLosassoVCycleHierarchySource {
  /** Finest first; every level uses the same closed-form axis-face operator. */
  readonly levels: readonly OctreeLosassoVCycleLevelSource[];
  readonly transfers: readonly OctreeLosassoVCycleTransferSource[];
  /**
   * Packed, publication-owned view of levels L1..Ln.  Keeping this view current
   * while hierarchy coefficients are published lets one workgroup execute the
   * complete sub-L0 cycle without copying geometry on every MGPCG iteration.
   */
  readonly fusedSubL0?: {
    readonly arena: GPUBuffer;
    readonly rowCapacity: number;
    readonly faceCapacity: number;
    readonly transitionStrideWords: number;
    readonly controlOffsetWords: number;
    readonly rowOffsetsOffsetWords: number;
    readonly rowFacesOffsetWords: number;
    readonly facesOffsetWords: number;
    readonly parentsOffsetWords: number;
    readonly childOffsetsOffsetWords: number;
    readonly childListOffsetWords: number;
    /** Safe per-level vector bounds derived from dyadic domain cell counts. */
    readonly levelRowCapacities: readonly number[];
  };
}

/**
 * Plain first-order V-cycle contract used directly as M^-1 by warm-started CG.
 * It refines the shared solver interface with negative capabilities: there is
 * no boundary-band operator, second-order smoother, or alternate level
 * operator to accidentally select.
 */
export interface OctreeLosassoFirstOrderVCycle extends OctreeFirstOrderSPDVCycle {
  readonly backend: "losasso";
  readonly operatorFamily: "closed-form-axis-face";
  readonly levelOperator: "same-first-order-operator";
  readonly boundaryBandSmoother: "absent";
  readonly schedule: typeof OCTREE_LOSASSO_VCYCLE_SCHEDULE;
  readonly hierarchy: OctreeLosassoVCycleHierarchySource;
}

export interface OctreeLosassoVCyclePlan {
  readonly levelRowCapacities: readonly number[];
  readonly levelCount: number;
  readonly preSweeps: 2;
  readonly postSweeps: 2;
  /** Down smooth/restrict, bottom solve, prolong/post smooth. */
  readonly correctionStageCount: number;
}

/** Author a fixed, symmetric schedule. No tolerance or adaptive tail exists. */
export function planOctreeLosassoVCycle(
  levelRowCapacities: readonly number[],
): OctreeLosassoVCyclePlan {
  if (levelRowCapacities.length < 1) throw new RangeError("Losasso V-cycle needs at least one level");
  for (const [level, rows] of levelRowCapacities.entries()) {
    if (!Number.isSafeInteger(rows) || rows < 1) {
      throw new RangeError(`Losasso V-cycle level ${level} capacity must be positive`);
    }
    if (level > 0 && rows > levelRowCapacities[level - 1]!) {
      throw new RangeError("Losasso V-cycle capacities must not grow toward the coarse grid");
    }
  }
  const levelCount = levelRowCapacities.length;
  return Object.freeze({
    levelRowCapacities: Object.freeze([...levelRowCapacities]),
    levelCount,
    preSweeps: 2 as const,
    postSweeps: 2 as const,
    correctionStageCount: 2 * (levelCount - 1) + 1,
  });
}

/** Fail early at backend construction rather than accepting a hybrid wrapper. */
export function assertOctreeLosassoVCycle(
  cycle: OctreeFirstOrderSPDVCycle,
): asserts cycle is OctreeLosassoFirstOrderVCycle {
  const candidate = cycle as Partial<OctreeLosassoFirstOrderVCycle>;
  if (cycle.operatorOrder !== 1 || cycle.isSymmetricPositiveDefinite !== true
    || cycle.convergenceTail !== "gpu-zero-indirect"
    || candidate.backend !== "losasso"
    || candidate.operatorFamily !== "closed-form-axis-face"
    || candidate.levelOperator !== "same-first-order-operator"
    || candidate.boundaryBandSmoother !== "absent"
    || candidate.schedule?.preRelaxations !== 2
    || candidate.schedule.postRelaxations !== 2
    || candidate.schedule.coarseSolve !== "fixed") {
    throw new Error("Losasso requires one plain symmetric first-order V-cycle with no hybrid boundary path");
  }
  const hierarchy = candidate.hierarchy;
  if (!hierarchy || hierarchy.levels.length < 1
    || hierarchy.transfers.length !== hierarchy.levels.length - 1) {
    throw new Error("Losasso V-cycle hierarchy/transfer publication is incomplete");
  }
}

/**
 * Brand a raw first-order hierarchy at construction without wrapping it in the
 * retired Section 4.3 boundary smoother. Method closures retain the original
 * object's receiver, so existing plain SPGrid V-cycles can cross the seam.
 */
export function createOctreeLosassoVCycleContract(
  cycle: OctreeFirstOrderSPDVCycle,
  hierarchy: OctreeLosassoVCycleHierarchySource,
): OctreeLosassoFirstOrderVCycle {
  if (cycle.operatorOrder !== 1 || cycle.isSymmetricPositiveDefinite !== true
    || cycle.convergenceTail !== "gpu-zero-indirect") {
    throw new Error("Losasso preconditioning requires a fixed SPD first-order V-cycle");
  }
  if (hierarchy.levels.length < 1 || hierarchy.transfers.length !== hierarchy.levels.length - 1) {
    throw new Error("Losasso V-cycle hierarchy/transfer publication is incomplete");
  }
  const result: OctreeLosassoFirstOrderVCycle = {
    operatorOrder: 1,
    isSymmetricPositiveDefinite: true,
    convergenceTail: "gpu-zero-indirect",
    allocatedBytes: cycle.allocatedBytes,
    encodedCorrectionDispatchCount: cycle.encodedCorrectionDispatchCount,
    encodedSetupDispatchCount: cycle.encodedSetupDispatchCount,
    encodedPassTransitionCount: cycle.encodedPassTransitionCount,
    smootherContract: cycle.smootherContract,
    backend: "losasso",
    operatorFamily: "closed-form-axis-face",
    levelOperator: "same-first-order-operator",
    boundaryBandSmoother: "absent",
    schedule: OCTREE_LOSASSO_VCYCLE_SCHEDULE,
    hierarchy,
    encodeSetup: (broker, input) => cycle.encodeSetup(broker, input),
    encodeCorrection: (broker, input) => cycle.encodeCorrection(broker, input),
    ...(cycle.encodeCorrectionGate && cycle.encodeCorrectionBody ? {
      encodeCorrectionGate: (pass: GPUComputePassEncoder, input: {
        readonly rhs: GPUBuffer; readonly correction: GPUBuffer;
        readonly solverControl: GPUBuffer; readonly rowCount: GPUBuffer;
      }) => cycle.encodeCorrectionGate!(pass, input),
      encodeCorrectionBody: (broker: PassBroker, input: {
        readonly rhs: GPUBuffer; readonly correction: GPUBuffer;
        readonly solverControl: GPUBuffer; readonly rowCount: GPUBuffer;
      }) => cycle.encodeCorrectionBody!(broker, input),
    } : {}),
  };
  assertOctreeLosassoVCycle(result);
  return Object.freeze(result);
}
