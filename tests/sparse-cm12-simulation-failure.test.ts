import assert from "node:assert/strict";
import test from "node:test";
import { SimulationFailureError } from "../lib/core/simulation-failure";
import { CM12_FAILURE_WORDS, cm12FailureKernelId, decodeCM12SimulationFailure } from "../lib/methods/adaptive-mass/sparse-cm12-simulation-failure";

test("failure receipts preserve raw provenance through JSON and reject incomplete reads", () => {
  const words = new Uint32Array(CM12_FAILURE_WORDS);
  assert.equal(decodeCM12SimulationFailure(words), undefined);
  words.set([1, 1, cm12FailureKernelId("forceFaces"), 42, 17, 123, 900, 2, 384, 0]);
  const failure = decodeCM12SimulationFailure(words)!;
  assert.equal(failure.kernel, "forceFaces");
  assert.equal(failure.code, "INCIDENCE_RANGE");
  assert.deepEqual(failure.operands, [900, 2, 384, 0]);
  assert.deepEqual(JSON.parse(JSON.stringify(failure)), failure);
  assert.match(new SimulationFailureError(failure).message, /HALTED.*frame=42.*generation=17.*owner=123/);
  assert.throws(() => decodeCM12SimulationFailure(words.slice(1)), /Incomplete/);
});

test("transport fault operands decode as floats while preserving exact raw bits", () => {
  const words = new Uint32Array(CM12_FAILURE_WORDS);
  words.set([1, 4, cm12FailureKernelId("gatherTransport"), 2, 3, 4]);
  words.set(new Uint32Array(new Float32Array([-0.25, Infinity, 0, 0]).buffer), 6);
  const failure = decodeCM12SimulationFailure(words)!;
  assert.deepEqual(failure.operands, [-0.25, Infinity, 0, 0]);
  assert.equal(failure.operandNames?.[0], "rawDensity");
  assert.equal(JSON.parse(JSON.stringify(failure)).rawWords[7], 0x7f800000);
});
