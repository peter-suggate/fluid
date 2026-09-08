import assert from "node:assert/strict";
import test from "node:test";
import { CM12_FAILURE_WORDS, cm12FailureKernelId, decodeCM12SimulationFailure } from
  "../lib/methods/adaptive-mass/sparse-cm12-simulation-failure";

test("retained integral faults expose density mismatch without losing stage or raw bits", () => {
  const words = new Uint32Array(CM12_FAILURE_WORDS);
  words.set([1, 6, cm12FailureKernelId("compileRetainedDensityNativeIntegrals"), 8, 12, 31, 3]);
  words.set(new Uint32Array(new Float32Array([0.25, 0.5, 0.75]).buffer), 7);
  const failure = decodeCM12SimulationFailure(words)!;
  assert.equal(failure.code, "RETAINED_DENSITY_INTEGRAL");
  assert.deepEqual(failure.operands, [3, 0.25, 0.5, 0.75]);
  assert.deepEqual(failure.operandNames, ["stage", "density", "targetDensity", "previousDensity"]);
  assert.deepEqual(failure.rawWords, [...words]);
});

test("rigid displacement receipts preserve integer hop distances and floating amounts", () => {
  const words = new Uint32Array(CM12_FAILURE_WORDS);
  words.set([1, 6, cm12FailureKernelId("validateRetainedRigidDisplacement"), 2, 3, 17, 4]);
  words.set(new Uint32Array(new Float32Array([.625, 0]).buffer), 7);
  words[9] = 0xffffffff;
  const route = decodeCM12SimulationFailure(words)!;
  assert.deepEqual(route.operands, [4, .625, 0, 0xffffffff]);
  assert.deepEqual(route.operandNames, ["stage", "density", "openFraction", "hopDistance"]);
  words[6] = 5;
  words[8] = 64;
  words[9] = 0;
  const packet = decodeCM12SimulationFailure(words)!;
  assert.deepEqual(packet.operands, [5, .625, 64, 0]);
  assert.deepEqual(packet.operandNames, ["stage", "packetAmount", "hopDistance", "reserved"]);
});
