import assert from "node:assert/strict";
import test from "node:test";

import {
  admitSparseReflectionOrbits,
  advanceSparseReflectionAdmission,
} from "../sparse-symmetric-admission";

test("topology admission does not split a three-axis reflection orbit", () => {
  const dimensions = [8, 6, 4] as const;
  const candidates = Array.from({ length: 8 }, (_, mask) => {
    const coordinate = [
      mask & 1 ? 6 : 0,
      mask & 2 ? 4 : 0,
      mask & 4 ? 2 : 0,
    ] as const;
    return { key: mask, coordinate, spanBricks: 2, priority: 7, transition: "8/4" };
  });
  assert.deepEqual([...admitSparseReflectionOrbits(candidates, dimensions, 7)], [],
    "a hard budget smaller than the orbit must defer the whole orbit");
  assert.deepEqual([...admitSparseReflectionOrbits(candidates, dimensions, 8)].sort((a, b) => a - b),
    [0, 1, 2, 3, 4, 5, 6, 7]);
});

test("topology admission groups only reflected bricks present in the request", () => {
  const candidates = [
    { key: 10, coordinate: [1, 0, 0] as const, priority: 9, transition: "8/4" },
    { key: 11, coordinate: [4, 0, 0] as const, priority: 9, transition: "8/4" },
    { key: 12, coordinate: [2, 0, 0] as const, priority: 1, transition: "8/4" },
  ];
  assert.deepEqual([...admitSparseReflectionOrbits(candidates, [6, 1, 1], 2)]
    .sort((a, b) => a - b), [10, 11]);
});

test("selection credit carries a deferred orbit across actual admission steps", () => {
  const candidates = [
    { key: 1, coordinate: [0, 0, 0] as const, priority: 1, transition: "8/4" },
    { key: 2, coordinate: [1, 0, 0] as const, priority: 1, transition: "8/4" },
  ];
  const first = advanceSparseReflectionAdmission(candidates, [2, 1, 1], 0, 1);
  assert.equal(first.admitted.size, 0); assert.equal(first.remainingCredit, 1);
  const second = advanceSparseReflectionAdmission(candidates, [2, 1, 1],
    first.remainingCredit, 1);
  assert.deepEqual([...second.admitted].sort((a, b) => a - b), [1, 2]);
  assert.equal(second.remainingCredit, 0);
});

test("zero selection increment neither invents credit nor admits work", () => {
  const result = advanceSparseReflectionAdmission([
    { key: 1, coordinate: [0, 0, 0], priority: 1, transition: "8/4" },
  ], [1, 1, 1], 0, 0);
  assert.equal(result.admitted.size, 0); assert.equal(result.remainingCredit, 0);
});

test("priority and transition mismatches are not coupled into one orbit", () => {
  const candidates = [
    { key: 1, coordinate: [0, 0, 0] as const, priority: 2, transition: "8/4" },
    { key: 2, coordinate: [1, 0, 0] as const, priority: 1, transition: "8/4" },
    { key: 3, coordinate: [0, 1, 0] as const, priority: 2, transition: "4/2" },
  ];
  const result = advanceSparseReflectionAdmission(candidates, [2, 2, 1], 0, 1);
  assert.deepEqual([...result.admitted], [1]);
  assert.equal(result.remainingCredit, 0);
});
