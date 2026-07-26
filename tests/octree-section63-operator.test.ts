import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_SECTION63_COEFFICIENT_CHANNELS,
  applyResolvedPressureOracle,
  applySection63Operator,
  assembleSection63Dense,
  buildSection63Operator,
  estimateSection63Bandwidth,
  type Section63Contact,
  type Section63Leaf,
} from "../lib/octree-section63-operator";

const close = (actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 1e-12) => {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(Math.abs(actual[index]! - expected[index]!) <= tolerance,
      `value ${index}: ${actual[index]} != ${expected[index]}`);
  }
};

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let result = 0;
  for (let index = 0; index < a.length; index += 1) result += a[index]! * b[index]!;
  return result;
};

const mixedLeaves: readonly Section63Leaf[] = Object.freeze([
  { origin: [0, 0, 0], size: 2, anchor: 0.7 },
  { origin: [2, 0, 0], size: 1, anchor: 0.2 },
  { origin: [2, 1, 0], size: 1, anchor: 0.1 },
]);
const mixedContacts: readonly Section63Contact[] = Object.freeze([
  { negative: 0, positive: 1, coefficient: 1.25 },
  { negative: 0, positive: 2, coefficient: 0.75 },
  { negative: 1, positive: 2, coefficient: 0.4 },
]);

test("Section 6.3 mixed 2:1 E^T B E is differential-equal to resolved rows", () => {
  const operator = buildSection63Operator(mixedLeaves, mixedContacts);
  assert.equal(operator.coefficientChannels, OCTREE_SECTION63_COEFFICIENT_CHANNELS);
  assert.equal(operator.storedNeighbourIndices, 0);
  assert.ok(operator.levels[0]!.flags.filter((flag) => (flag & 2) !== 0).length === 2,
    "the coarse leaf must propagate into one ghost per fine contact cell");
  for (const input of [
    [0, 0, 0], [1, 1, 1], [0.25, -1.5, 2.75], [-7, 0.125, 3],
  ]) {
    close(applySection63Operator(operator, input),
      applyResolvedPressureOracle(mixedLeaves, mixedContacts, input));
  }
});

test("Section 6.3 transfer is symmetric and positive definite when anchored", () => {
  const operator = buildSection63Operator(mixedLeaves, mixedContacts);
  const matrix = assembleSection63Dense(operator), n = operator.leafCount;
  for (let row = 0; row < n; row += 1) for (let column = 0; column < n; column += 1) {
    assert.ok(Math.abs(matrix[row * n + column]! - matrix[column * n + row]!) < 1e-12);
  }
  for (const vector of [[1, 0, 0], [1, -2, 3], [-0.5, 4, 1.25]]) {
    const energy = dot(vector, applySection63Operator(operator, vector));
    assert.ok(energy > 0, `anchored Section 6.3 energy must be positive, got ${energy}`);
  }
});

test("Section 6.3 unanchored operator preserves constants across a 2:1 interface", () => {
  const leaves = mixedLeaves.map((leaf) => ({ ...leaf, anchor: 0 }));
  const operator = buildSection63Operator(leaves, mixedContacts);
  close(applySection63Operator(operator, [3.5, 3.5, 3.5]), [0, 0, 0]);
});

test("Section 6.3 coefficients resolve neighbours from coordinates and pages only", () => {
  const operator = buildSection63Operator(mixedLeaves, mixedContacts);
  for (const level of operator.levels) {
    assert.equal(level.coefficients.length,
      level.coordinates.length * OCTREE_SECTION63_COEFFICIENT_CHANNELS);
    assert.equal(level.pages.length, level.coordinates.length);
    assert.ok(level.pages.every((page) => Number.isSafeInteger(page)));
  }
  assert.equal("neighbours" in operator, false);
  assert.equal("neighborRows" in operator, false);
});

test("large Section 6.3 bandwidth lane exceeds cache and exposes gather traffic", () => {
  const estimate = estimateSection63Bandwidth(2_000_000, 18);
  assert.ok(estimate.cachePressureBytes.resolvedRows > 512 * 1024 * 1024,
    "the lane must exceed M1 Max system-level cache by a wide margin");
  assert.ok(estimate.cachePressureBytes.catalogTable < 128 * 1024,
    "the immutable 19-channel catalog must remain cache resident");
  assert.ok(estimate.byteReductionRatio > 2.5,
    `expected a visible gather-vs-stream separation, got ${estimate.byteReductionRatio}`);
  assert.equal(estimate.cachePressureBytes.pageValues, 8 * 8 * 4 * 4);
});

test("Section 6.3 rejects non-2:1 contacts instead of manufacturing a fallback", () => {
  assert.throws(() => buildSection63Operator([
    { origin: [0, 0, 0], size: 4 }, { origin: [4, 0, 0], size: 1 },
  ], [{ negative: 0, positive: 1, coefficient: 1 }]), /violates 2:1 grading/);
});
