import assert from "node:assert/strict";
import test from "node:test";
import { hermiteCell, initializeHermiteLine, quarticDerivative, quarticValue,
  sampleHermiteLine, translateHermiteLine, type ConservativeHermiteLine,
} from "../tools/implicit-density/conservative-hermite-oracle";

const tau = 2 * Math.PI;
function wave(n: number): ConservativeHermiteLine {
  return initializeHermiteLine(n, 1, x => .5 + .3 * Math.sin(tau * x),
    x => .3 * tau * Math.cos(tau * x),
    (a, b) => .5 * (b - a) + .3 / tau * (Math.cos(tau * a) - Math.cos(tau * b)));
}
const total = (line: ConservativeHermiteLine) => line.amount.reduce((a, b) => a + b, 0);

test("current-field transport preserves shared value/derivative and mass through a complete translated orbit", () => {
  const errors: number[] = [], slopeErrors: number[] = [];
  for (const n of [8, 16, 32]) {
    let line = wave(n); const initialAmount = total(line), steps = 3 * n;
    for (let step = 0; step < steps; step++) {
      line = translateHermiteLine(line, 1 / steps);
      assert.ok(Math.abs(total(line) - initialAmount) < 2e-13, "one field supplies both motion and conservative intervals");
      for (let cell = 0; cell < n; cell++) {
        const left = hermiteCell(line, cell), right = hermiteCell(line, cell + 1);
        assert.ok(Math.abs(quarticValue(left, 1) - quarticValue(right, 0)) < 2e-13);
        assert.ok(Math.abs(quarticDerivative(left, 1) - quarticDerivative(right, 0)) * n < 2e-11,
          "normal derivative must agree, not only the value");
      }
    }
    let error = 0, slopeError = 0;
    for (let i = 0; i <= 4096; i++) {
      const x = i / 4096, [q, derivative] = sampleHermiteLine(line, x);
      assert.ok(q >= 0 && q <= 1, "this smooth, resolved fixture needs no clipping");
      error = Math.max(error, Math.abs(q - (.5 + .3 * Math.sin(tau * x))));
      slopeError = Math.max(slopeError, Math.abs(derivative - .3 * tau * Math.cos(tau * x)));
    }
    errors.push(error); slopeErrors.push(slopeError);
  }
  // Analytic convergence, not comparison to an earlier implementation's output.
  assert.ok(errors[0]! / errors[1]! > 24 && errors[1]! / errors[2]! > 24, JSON.stringify(errors));
  assert.ok(slopeErrors[0]! / slopeErrors[1]! > 12 && slopeErrors[1]! / slopeErrors[2]! > 12, JSON.stringify(slopeErrors));
  assert.ok(errors[2]! < 3e-8 && slopeErrors[2]! < 2e-6);
});

test("fractional translation transports the field while equal global mass cannot identify its position", () => {
  const initial = wave(16), displacement = .137;
  const moved = translateHermiteLine(initial, displacement);
  assert.ok(Math.abs(total(moved) - total(initial)) < 1e-14);
  let actualError = 0, frozenError = 0;
  for (let i = 0; i < 1024; i++) {
    const x = i / 1024, expected = .5 + .3 * Math.sin(tau * (x - displacement));
    actualError = Math.max(actualError, Math.abs(sampleHermiteLine(moved, x)[0] - expected));
    frozenError = Math.max(frozenError, Math.abs(sampleHermiteLine(initial, x)[0] - expected));
  }
  assert.ok(actualError < 1e-6, String(actualError));
  assert.ok(frozenError > .24, "a mass-only acceptance would miss a large spatial error");
  const negative = translateHermiteLine(initial, displacement - 7);
  for (let i = 0; i < initial.value.length; i++) {
    assert.ok(Math.abs(moved.amount[i]! - negative.amount[i]!) < 1e-13,
      "periodic long traces must use the same physical departure interval");
  }
});

test("unresolved sharp data exposes the quartic positivity limit; no clipping disguises the failed representation", () => {
  // A positive monotone thin layer can have endpoint values 1/0, zero endpoint
  // slopes and mean 0.1. Those data force this quartic to undershoot: smooth
  // continuity and M0 alone do not establish an admissible density profile.
  const unresolved: ConservativeHermiteLine = { length: 2,
    value: Float64Array.from([1, 0]), derivative: new Float64Array(2),
    amount: Float64Array.from([.1, .1]) };
  assert.equal(quarticValue(hermiteCell(unresolved, 0), .5), -.25);
  assert.equal(total(unresolved), .2);
  // This is a negative research control. Production admission would have to
  // reject/refine or carry the sharp branch, not clamp away the negative mass.
});
