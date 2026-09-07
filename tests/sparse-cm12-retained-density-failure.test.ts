import assert from "node:assert/strict";
import test from "node:test";
import { CM12_FAILURE_WORDS, cm12FailureKernelId, decodeCM12SimulationFailure } from
  "../lib/methods/adaptive-mass/sparse-cm12-simulation-failure";

test("retained integral faults expose density mismatch without losing stage or raw bits", () => {
  const words = new Uint32Array(CM12_FAILURE_WORDS);
  words.set([1, 6, cm12FailureKernelId("compileRetainedDensityNativeIntegrals"), 8, 12, 31, 3]);
  words.set(new Uint32Array(new Float32Array([0.25, 0.5]).buffer), 7);
  const failure = decodeCM12SimulationFailure(words)!;
  assert.equal(failure.code, "RETAINED_DENSITY_INTEGRAL");
  assert.deepEqual(failure.operands, [3, 0.25, 0.5, 0]);
  assert.deepEqual(failure.operandNames, ["stage", "density", "targetDensity", "reserved"]);
  assert.deepEqual(failure.rawWords, [...words]);
});
