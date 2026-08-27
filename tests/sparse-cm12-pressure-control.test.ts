import assert from "node:assert/strict";
import test from "node:test";

import { adaptiveMassMethod } from
  "../lib/methods/adaptive-mass/method";
import {
  adaptiveMassPressureIterationReadout,
  sparseCM12Stage,
} from "../lib/methods/adaptive-mass/sparse-cm12-stages";
import {
  SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE,
  SPARSE_CM12_PRESSURE_TRUE_RESIDUAL_CADENCE,
  sparseCM12PressureIterations,
  sparseCM12PressureIterationsFromReceipt,
  sparseCM12PressureRelativeTolerance,
} from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

test("Sparse CM12 exposes fixed eight-iteration residual blocks", () => {
  assert.equal(SPARSE_CM12_PRESSURE_TRUE_RESIDUAL_CADENCE, 8);
  assert.equal(SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE, 1e-3);

  const spec = adaptiveMassMethod.params.find((candidate) =>
    candidate.key === "pressureRelativeTolerance");
  assert.equal(spec?.kind, "number");
  if (spec?.kind !== "number") return;
  assert.equal(spec.default, 1e-3);
  assert.equal(spec.max, 1);
  assert.equal(sparseCM12PressureIterations(25), 24);
  assert.equal(sparseCM12PressureIterations(29), 32);
});

test("Sparse CM12 accepts experimental residuals beyond 0.1", () => {
  assert.equal(sparseCM12PressureRelativeTolerance(0), 0);
  assert.equal(sparseCM12PressureRelativeTolerance(0.5), 0.5);
  assert.equal(sparseCM12PressureRelativeTolerance(2), 1);
});

test("a prior frame cannot truncate the next encoded pressure ceiling", () => {
  assert.equal(sparseCM12PressureIterationsFromReceipt(64, 1e-3), 64);
  assert.equal(sparseCM12PressureIterationsFromReceipt(64, 0,
    { executed: 16, encoded: 64 }), 64);
  assert.equal(sparseCM12PressureIterationsFromReceipt(64, 1e-3,
    { executed: 16, encoded: 64 }), 64);
  assert.equal(sparseCM12PressureIterationsFromReceipt(64, 1e-3,
    { executed: 24, encoded: 32 }), 64);
  assert.equal(sparseCM12PressureIterationsFromReceipt(64, 1e-3,
    { executed: 32, encoded: 32 }), 64);
  assert.equal(sparseCM12PressureIterationsFromReceipt(48, 1e-3,
    { executed: 32, encoded: 32 }), 48);
});

test("the SIM frame pressure stage shows executed and gated iterations", () => {
  assert.equal(adaptiveMassPressureIterationReadout({
    pressureIterationsExecuted: 24,
    pressureIterationsEncoded: 64,
  }, 64), "24 / 64");
  assert.equal(adaptiveMassPressureIterationReadout(undefined, 64),
    "— / 64");

  const readout = sparseCM12Stage("pressure-solve").controls?.find(
    (control) => control.kind === "readout"
      && control.label === "Iterations executed / encoded",
  );
  assert.equal(readout?.kind, "readout");
});
