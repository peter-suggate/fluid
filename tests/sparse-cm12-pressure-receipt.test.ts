import assert from "node:assert/strict";
import test from "node:test";
import {
  SPARSE_CM12_PRESSURE_RECEIPT_VERSION,
  assertSparseCM12PressureReceipt,
  sparseCM12PressureResidualDrift,
  type SparseCM12PressureReceipt,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-receipt";

function receipt(): SparseCM12PressureReceipt {
  return {
    version: SPARSE_CM12_PRESSURE_RECEIPT_VERSION,
    snapshot: "manufactured",
    solver: "gpu-sparse-pcg",
    topology: {
      denseCellCount: 64, acceptedCellCount: 16, acceptedRowCount: 24,
      liquidCellCount: 8, activePressureRowCount: 12,
      effectiveOffDiagonalCount: 14,
      rows: { regular: 8, "brick-face": 2, "mixed-seam": 1, "sparse-air": 1 },
      rowDegreeHistogram: { "1": 2, "2": 10 }, maximumRowDegree: 2,
    },
    theta: { minimum: 0.25, logarithmicHistogram: { "-2": 1 }, clampCount: 0 },
    components: {
      count: 1, anchored: 1, unanchored: 0,
      compatibilityCorrectionMaximum: 0, incompatible: false,
    },
    seed: {
      kind: "warm",
      trueInitial: { relativeL2: 0.1, maximum: 0.2,
        maximumDivergenceEquivalent_s: 0.2 },
      zeroSeedTrueInitial: { relativeL2: 1, maximum: 2,
        maximumDivergenceEquivalent_s: 2 },
      newlyLiquidCellCount: 0, fallbackToZeroCellCount: 0,
      physicalScaleRatio: 1, rescaled: false,
    },
    convergence: {
      converged: true, reason: "tolerance", encodedIterationCeiling: 128,
      executedIterations: 16, firstToleranceCrossingIteration: 16,
      gatedTailIterations: 112, spmvApplications: 19,
      recursiveRelativeL2: 9e-6,
      trueFinal: { relativeL2: 1e-5, maximum: 2e-5,
        maximumDivergenceEquivalent_s: 2e-5 },
      recursiveToTrueRatio: 0.9, residualDrift: false,
    },
    projection: { divergenceVolumeL2_s: 8e-6, divergenceMaximum_s: 2e-5 },
    timing_ms: {
      topology: 0.1, rhs: 0.1, solve: 1, projection: 0.1, diagnostics: 0.1,
      total: 1.4, method: "gpu-timestamp",
    },
    memoryBytes: {
      pressureState: 1024, effectiveEdges: 112, reductionPartials: 16,
      hierarchy: 0, total: 1152,
    },
  };
}

test("pressure receipts distinguish recursive and true convergence", () => {
  const value = receipt();
  assert.doesNotThrow(() => assertSparseCM12PressureReceipt(value));
  assert.equal(sparseCM12PressureResidualDrift(1e-8, 1e-5), true);
  assert.equal(sparseCM12PressureResidualDrift(9e-6, 1e-5), false);
});

test("pressure receipts reject inconsistent encoded tails", () => {
  const value = receipt();
  const invalid = {
    ...value,
    convergence: { ...value.convergence, gatedTailIterations: 111 },
  };
  assert.throws(() => assertSparseCM12PressureReceipt(invalid), /gated-tail/);
});
