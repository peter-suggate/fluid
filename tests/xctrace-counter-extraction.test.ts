import assert from "node:assert/strict";
import test from "node:test";
import {
  CounterRowSelector,
  makeCounterExtractionPolicy,
  RETAINED_GPU_COUNTERS,
} from "../tools/profile-mini-dam-xctrace";

test("counter extraction reduces rows by at least 100x without touching stage tables", () => {
  const counters = new Map<string, string>();
  const required = [...RETAINED_GPU_COUNTERS];
  for (let index = 0; index < 31; index += 1) {
    counters.set(String(index), required[index] ?? `unused counter ${index}`);
  }
  const policy = makeCounterExtractionPolicy(counters, 100);
  assert.equal(policy.retainedCounterCount, 5);
  assert.equal(policy.timestampStride, 17);

  const selector = new CounterRowSelector(policy);
  let source = 0;
  let retained = 0;
  for (let timestamp = 0; timestamp < 170; timestamp += 1) {
    for (let counter = 0; counter < 31; counter += 1) {
      for (let ring = 0; ring < 4; ring += 1) {
        source += 1;
        if (selector.keep({
          timestamp: String(timestamp),
          "counter-id": String(counter),
          "ring-buffer-index": String(ring),
        })) retained += 1;
      }
    }
  }
  assert.equal(retained, 200);
  assert.ok(source / retained >= 100, `${source / retained}x retained-row reduction`);
});

test("counter reduction 1 retains every counter at every timestamp", () => {
  const counters = new Map([["0", "Compute Occupancy"], ["1", "unused"]]);
  const policy = makeCounterExtractionPolicy(counters, 1);
  assert.equal(policy.timestampStride, 1);
  assert.deepEqual([...policy.retainedCounterIds], ["0", "1"]);
});
