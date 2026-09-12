import assert from "node:assert/strict";
import test from "node:test";

import { addSliceSourceCompensated } from "./slice-dynamic-remap";

const f = Math.fround;

function accumulate(values: readonly number[]): readonly [number, number] {
  let total = 0, compensation = 0;
  for (const value of values) {
    [total, compensation] = addSliceSourceCompensated(value, total, compensation);
  }
  return [total, compensation];
}

test("source ledger retains small signed increments across outer-frame carries", () => {
  const values = [16_777_216, 1, 1, -16_777_216, -0.25, 0.125, 0.125];
  const [total, compensation] = accumulate(values);
  assert.equal(total, 2);
  assert.ok(Object.is(compensation, -0));

  let naive = 0;
  for (const value of values) naive = f(naive + f(value));
  assert.equal(naive, 0);
});

test("source ledger preserves a signed residual when a rounded total is published", () => {
  const [total, compensation] = addSliceSourceCompensated(-8.824240684509277,
    111.5223617553711, 0);
  assert.equal(total, 102.6981201171875);
  assert.equal(compensation, -9.5367431640625e-7);
  const [next, nextCompensation] = addSliceSourceCompensated(-8.824240684509277,
    total, compensation);
  assert.equal(next, 93.8738784790039);
  assert.equal(nextCompensation, -0.0000019073486328125);
});
