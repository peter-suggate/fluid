/**
 * Versioned, JSON-safe receipt for a frozen Sparse CM12 pressure solve.
 *
 * This is intentionally independent of WebGPU objects.  Dawn benchmarks,
 * the CPU composite oracle, and saved benchmark artifacts all publish the
 * same shape so a recursively updated CG residual can never be mistaken for
 * the convergence authority.
 */

export const SPARSE_CM12_PRESSURE_RECEIPT_VERSION = 1 as const;

export type SparseCM12PressureRowClass =
  | "regular"
  | "brick-face"
  | "mixed-seam"
  | "sparse-air";

export interface SparseCM12PressureResidualReceipt {
  readonly relativeL2: number;
  readonly maximum: number;
  /** Maximum finite-volume equation residual divided by cell volume, in 1/s. */
  readonly maximumDivergenceEquivalent_s: number;
}

export interface SparseCM12PressureReceipt {
  readonly version: typeof SPARSE_CM12_PRESSURE_RECEIPT_VERSION;
  readonly snapshot: string;
  readonly solver: "cpu-composite-oracle" | "gpu-sparse-pcg" | "gpu-sparse-mgpcg";
  readonly topology: {
    readonly denseCellCount: number;
    readonly acceptedCellCount: number;
    readonly acceptedRowCount: number;
    readonly liquidCellCount: number;
    readonly activePressureRowCount: number;
    readonly effectiveOffDiagonalCount: number;
    readonly rows: Readonly<Record<SparseCM12PressureRowClass, number>>;
    /** Degree -> active-row count. Numeric keys stay stable when serialized. */
    readonly rowDegreeHistogram: Readonly<Record<string, number>>;
    readonly maximumRowDegree: number;
  };
  readonly theta: {
    readonly minimum: number;
    /** Floor(log2(theta)) -> active cut-row count. */
    readonly logarithmicHistogram: Readonly<Record<string, number>>;
    readonly clampCount: number;
  };
  readonly components: {
    readonly count: number;
    readonly anchored: number;
    readonly unanchored: number;
    readonly compatibilityCorrectionMaximum: number;
    readonly incompatible: boolean;
  };
  readonly seed: {
    readonly kind: "zero" | "warm" | "transferred";
    readonly trueInitial: SparseCM12PressureResidualReceipt;
    readonly zeroSeedTrueInitial?: SparseCM12PressureResidualReceipt;
    readonly newlyLiquidCellCount: number;
    readonly fallbackToZeroCellCount: number;
    readonly physicalScaleRatio: number;
    readonly rescaled: boolean;
  };
  readonly convergence: {
    readonly converged: boolean;
    readonly reason: "tolerance" | "iteration-cap" | "curvature" | "nonfinite";
    readonly encodedIterationCeiling: number;
    readonly executedIterations: number;
    readonly firstToleranceCrossingIteration: number | null;
    readonly gatedTailIterations: number;
    readonly spmvApplications: number;
    readonly recursiveRelativeL2: number;
    readonly trueFinal: SparseCM12PressureResidualReceipt;
    readonly recursiveToTrueRatio: number;
    readonly residualDrift: boolean;
  };
  readonly projection: {
    readonly divergenceVolumeL2_s: number;
    readonly divergenceMaximum_s: number;
  };
  readonly timing_ms: {
    readonly topology: number;
    readonly rhs: number;
    readonly solve: number;
    readonly projection: number;
    readonly diagnostics: number;
    readonly total: number;
    readonly method: "cpu-wall" | "gpu-timestamp" | "serialized-queue-wall";
  };
  readonly memoryBytes: {
    readonly pressureState: number;
    readonly effectiveEdges: number;
    readonly reductionPartials: number;
    readonly hierarchy: number;
    readonly total: number;
  };
}

const finiteNonnegative = (value: number): boolean =>
  Number.isFinite(value) && value >= 0;

/** Fail fast before a receipt is written to a benchmark artifact. */
export function assertSparseCM12PressureReceipt(
  receipt: SparseCM12PressureReceipt,
): void {
  if (receipt.version !== SPARSE_CM12_PRESSURE_RECEIPT_VERSION) {
    throw new Error(`unsupported Sparse CM12 pressure receipt version ${receipt.version}`);
  }
  if (!receipt.snapshot) throw new Error("pressure receipt snapshot is empty");
  const counts = [
    receipt.topology.denseCellCount,
    receipt.topology.acceptedCellCount,
    receipt.topology.acceptedRowCount,
    receipt.topology.liquidCellCount,
    receipt.topology.activePressureRowCount,
    receipt.topology.effectiveOffDiagonalCount,
    receipt.topology.maximumRowDegree,
    receipt.convergence.encodedIterationCeiling,
    receipt.convergence.executedIterations,
    receipt.convergence.gatedTailIterations,
    receipt.convergence.spmvApplications,
  ];
  if (counts.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new Error("pressure receipt contains an invalid count");
  }
  if (receipt.convergence.executedIterations
    > receipt.convergence.encodedIterationCeiling) {
    throw new Error("executed pressure iterations exceed the encoded ceiling");
  }
  if (receipt.convergence.gatedTailIterations
    !== receipt.convergence.encodedIterationCeiling
      - receipt.convergence.executedIterations) {
    throw new Error("pressure gated-tail count is inconsistent");
  }
  const scalars = [
    receipt.theta.minimum,
    receipt.components.compatibilityCorrectionMaximum,
    receipt.seed.physicalScaleRatio,
    receipt.convergence.recursiveRelativeL2,
    receipt.convergence.trueFinal.relativeL2,
    receipt.convergence.trueFinal.maximum,
    receipt.convergence.trueFinal.maximumDivergenceEquivalent_s,
    receipt.projection.divergenceVolumeL2_s,
    receipt.projection.divergenceMaximum_s,
    receipt.timing_ms.total,
    receipt.memoryBytes.total,
  ];
  if (scalars.some((value) => !finiteNonnegative(value))) {
    throw new Error("pressure receipt contains a non-finite or negative scalar");
  }
  if (receipt.convergence.converged
    && receipt.convergence.reason !== "tolerance") {
    throw new Error("a converged pressure receipt must stop on tolerance");
  }
  if (!receipt.convergence.converged
    && receipt.convergence.reason === "tolerance") {
    throw new Error("an unconverged pressure receipt cannot stop on tolerance");
  }
}

export function sparseCM12PressureResidualDrift(
  recursiveRelativeL2: number,
  trueRelativeL2: number,
  materialRatio = 4,
): boolean {
  if (!finiteNonnegative(recursiveRelativeL2)
    || !finiteNonnegative(trueRelativeL2)
    || !(materialRatio >= 1)) return true;
  if (trueRelativeL2 === 0) return false;
  return recursiveRelativeL2 * materialRatio < trueRelativeL2;
}
