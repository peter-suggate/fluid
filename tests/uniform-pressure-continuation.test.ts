import assert from "node:assert/strict";
import test from "node:test";
import { nextUniformPressureCorrection as next } from "../lib/methods/uniform/uniform-pressure-continuation";

test("useful V-cycle progress keeps loose accuracy and inexpensive corrections", () => {
  assert.deepEqual(next(0, 4, 3, 12, 40, 1), {cycle:1,accuracy:1});
  assert.deepEqual(next(0, 4, 3, 20, 40, 1), {cycle:1,accuracy:1});
  assert.deepEqual(next(1, 4, 3, 11, 30, 1), {cycle:2,accuracy:1});
});
test("stalls skip to Full-Cycles and tighten progressively", () => {
  assert.deepEqual(next(0, 4, 3, 30, 40, 1), {cycle:4,accuracy:0.1});
  assert.deepEqual(next(4, 4, 3, 25, 30, 0.1), {cycle:5,accuracy:0});
  assert.deepEqual(next(5, 4, 3, 11, 25, 0.1), {cycle:6,accuracy:0.1});
});
test("configured budgets remain bounded with either cycle type absent", () => {
  assert.deepEqual(next(3, 4, 3, 11, 30, 1), {cycle:4,accuracy:1});
  assert.deepEqual(next(0, 4, 0, 30, 40, 1), {cycle:1,accuracy:0.1});
  assert.deepEqual(next(0, 0, 3, 11, 30, 1), {cycle:1,accuracy:1});
  assert.equal(next(6, 4, 3, 11, 30, 1), undefined);
  assert.equal(next(0, 0, 1, 11, 30, 1), undefined);
});
