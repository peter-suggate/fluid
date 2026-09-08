import assert from "node:assert/strict";
import test from "node:test";
import { initializeHermiteLine, sampleHermiteLine, translateHermiteLine } from "../tools/implicit-density/conservative-hermite-oracle";
import { initializeTensorCSL4, sampleTensorCSL4, tensorAmount, tensorCellBernstein,
  tensorFaceIdentity, tensorFunctional, tensorRangeAdmission, tensorSlot, translateTensorCSL4,
  type AxisFunctional, type SeparableFactor, type TensorCSL4Field, type Triple,
} from "../tools/implicit-density/tensor-csl4-oracle";

const tau = 2 * Math.PI;
const one: SeparableFactor = { value: () => 1, derivative: () => 0, integral: (a, b) => b - a };
const sin: SeparableFactor = { value: x => Math.sin(tau * x), derivative: x => tau * Math.cos(tau * x),
  integral: (a, b) => (Math.cos(tau * a) - Math.cos(tau * b)) / tau };
const cos: SeparableFactor = { value: x => Math.cos(tau * x), derivative: x => -tau * Math.sin(tau * x),
  integral: (a, b) => (Math.sin(tau * b) - Math.sin(tau * a)) / tau };
function wave(n: number): TensorCSL4Field {
  return initializeTensorCSL4([n, n, n], [1, 1, 1], [
    { scale: .5, factors: [one, one, one] }, { scale: .1, factors: [sin, sin, sin] }, { scale: .07, factors: [cos, cos, cos] },
  ]);
}
function waveExact(p: Triple): readonly [number, number, number, number] {
  const s = p.map(x => Math.sin(tau * x)), c = p.map(x => Math.cos(tau * x));
  return [.5 + .1 * s[0]! * s[1]! * s[2]! + .07 * c[0]! * c[1]! * c[2]!,
    tau * (.1 * c[0]! * s[1]! * s[2]! - .07 * s[0]! * c[1]! * c[2]!),
    tau * (.1 * s[0]! * c[1]! * s[2]! - .07 * c[0]! * s[1]! * c[2]!),
    tau * (.1 * s[0]! * s[1]! * c[2]! - .07 * c[0]! * c[1]! * s[2]!)];
}
const near = (actual: number, expected: number, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) < tolerance,
  `actual ${actual}; expected ${expected}; absolute error ${Math.abs(actual - expected)}`);

test("full tensor dimensional reduction agrees with the independent 1D quartic, including every stored mixed moment", () => {
  let line = initializeHermiteLine(8, 1, x => .5 + .3 * sin.value(x), x => .3 * sin.derivative(x),
    (a, b) => .5 * (b - a) + .3 * sin.integral(a, b));
  let field = initializeTensorCSL4([8, 3, 4], [1, 2, 3], [
    { scale: .5, factors: [one, one, one] }, { scale: .3, factors: [sin, one, one] },
  ]);
  for (const displacement of [0, .137, .25, -.43]) {
    field = translateTensorCSL4(field, [displacement, .271, -.137]);
    line = translateHermiteLine(line, displacement);
    for (let z = 0; z < 12; z++) for (let y = 0; y < 9; y++) for (let x = 0; x < 24; x++) {
      const i = Math.floor(x / 3), type = x % 3;
      const expected = y % 3 === 1 || z % 3 === 1 ? 0
        : type === 0 ? line.value[i]! : type === 1 ? line.derivative[i]! / 8 : line.amount[i]! * 8;
      near(field.data[x + 24 * (y + 9 * z)]!, expected, 5e-13);
    }
    for (let i = 0; i < 64; i++) {
      const p: Triple = [(i + .37) / 64, .13 + i / 32, 1.31 + i / 64];
      const actual = sampleTensorCSL4(field, p), expected = sampleHermiteLine(line, p[0]);
      near(actual[0], expected[0], 5e-13); near(actual[1], expected[1], 5e-12);
      near(actual[2], 0, 3e-12); near(actual[3], 0, 3e-12);
    }
  }
});

test("all 27 translated mixed functionals equal exact departure measurements of the current tensor field", () => {
  const field = wave(3), shift: Triple = [.047, -.073, .119], next = translateTensorCSL4(field, shift), h = 1 / 3;
  let maximumError = 0;
  for (let z = 0; z < 3; z++) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) {
    const cell: Triple = [x, y, z];
    for (let tz = 0; tz < 3; tz++) for (let ty = 0; ty < 3; ty++) for (let tx = 0; tx < 3; tx++) {
      const types: Triple = [tx, ty, tz];
      let normalization = 1;
      const fs = cell.map((i, axis): AxisFunctional => {
        const lower = i * h - shift[axis]!;
        if (types[axis] === 2) { normalization /= h; return { kind: "integral", lower, upper: lower + h }; }
        if (types[axis] === 1) normalization *= h;
        return { kind: "point", position: lower, derivative: types[axis] === 1 };
      }) as [AxisFunctional, AxisFunctional, AxisFunctional];
      maximumError = Math.max(maximumError, Math.abs(next.data[tensorSlot(next, cell, types)]! - normalization * tensorFunctional(field, fs)));
    }
  }
  assert.ok(maximumError < 3e-13, String(maximumError));
  near(tensorAmount(next), tensorAmount(field), 2e-14);
  const fullBox = field.lengths.map(length => ({ kind: "integral" as const, lower: -.123, upper: length - .123 })) as [AxisFunctional, AxisFunctional, AxisFunctional];
  near(tensorFunctional(field, fullBox), tensorAmount(field), 5e-14);
  // Re-integrating each cell must recover its own conserved volume moment.
  for (let z = 0; z < 3; z++) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) {
    const fs = [x, y, z].map(i => ({ kind: "integral" as const, lower: i * h, upper: (i + 1) * h })) as [AxisFunctional, AxisFunctional, AxisFunctional];
    near(tensorFunctional(next, fs), next.data[tensorSlot(next, [x, y, z], [2, 2, 2])]! * h ** 3, 2e-14);
  }
});

test("quartic reproduction holds in interior cells, whereas a single volume bubble invents transverse structure", () => {
  const fourth: SeparableFactor = { value: x => x ** 4, derivative: x => 4 * x ** 3, integral: (a, b) => (b ** 5 - a ** 5) / 5 };
  const field = initializeTensorCSL4([3, 3, 3], [3, 3, 3], [
    { scale: .5, factors: [one, one, one] }, { scale: .1, factors: [fourth, one, one] },
  ]);
  for (const y of [0, .137, .5, .971, 1]) {
    const q = sampleTensorCSL4(field, [.5, y, .5]);
    near(q[0], .50625); near(q[1], .05); near(q[2], 0); near(q[3], 0);
  }
  // The rejected 64-jet + one cell-volume bubble candidate is NOT the tensor.
  // Its tricubic interpolant is .5+.1(2x^3-x^2), with mean residual .1/30.
  const beta = (t: number) => 30 * t * t * (1 - t) ** 2;
  const shortcut = (x: number, y: number, z: number) => .5 + .1 * (2 * x ** 3 - x ** 2) + .1 / 30 * beta(x) * beta(y) * beta(z);
  near(shortcut(.5, 0, .5), .5);
  near(shortcut(.5, .5, .5), .52197265625);
  assert.ok(Math.abs(shortcut(.5, 0, .5) - shortcut(.5, .5, .5)) > .02,
    "one cell mass bubble fails dimensional consistency even with positive smooth polynomial data");
});

test("mixed quartic products reproduce independent point derivatives and multi-cell box integrals", () => {
  const power = (degree: number): SeparableFactor => ({ value: x => x ** degree,
    derivative: x => degree * x ** (degree - 1), integral: (a, b) => (b ** (degree + 1) - a ** (degree + 1)) / (degree + 1) });
  const factors = [power(2), power(3), power(4)] as const;
  const field = initializeTensorCSL4([3, 3, 3], [3, 3, 3], [
    { scale: .3, factors: [one, one, one] }, { scale: .001, factors },
  ]);
  // Only interior cells: a global nonperiodic polynomial cannot match across
  // the unrelated periodic outer seam. All queried source pieces are exact.
  const p: Triple = [1.31, .47, 1.2], q = sampleTensorCSL4(field, p);
  near(q[0], .3 + .001 * p[0] ** 2 * p[1] ** 3 * p[2] ** 4);
  near(q[1], .002 * p[0] * p[1] ** 3 * p[2] ** 4);
  near(q[2], .003 * p[0] ** 2 * p[1] ** 2 * p[2] ** 4);
  near(q[3], .004 * p[0] ** 2 * p[1] ** 3 * p[2] ** 3);
  const lower: Triple = [.17, .38, .21], upper: Triple = [1.73, 1.57, 1.64];
  const box = lower.map((lo, axis) => ({ kind: "integral" as const, lower: lo, upper: upper[axis]! })) as [AxisFunctional, AxisFunctional, AxisFunctional];
  const volume = upper.reduce((product, hi, axis) => product * (hi - lower[axis]!), 1);
  const expected = .3 * volume + .001 * factors.reduce((product, f, axis) => product * f.integral(lower[axis]!, upper[axis]!), 1);
  near(tensorFunctional(field, box), expected, 2e-12);
  const mixed: [AxisFunctional, AxisFunctional, AxisFunctional] = [
    { kind: "point", position: p[0], derivative: true }, box[1], { kind: "point", position: p[2], derivative: true },
  ];
  near(tensorFunctional(field, mixed), .001 * factors[0].derivative(p[0]) * factors[1].integral(lower[1], upper[1]) * factors[2].derivative(p[2]), 2e-12);
});

test("Bernstein bounds reject unresolved saturated plane moments without clipping or changing mass", () => {
  // Positive clipped plane: q=1 below .015, a linear ramp to zero at .035.
  // In cell [0,.25], both endpoint slopes vanish and its exact mean is .1.
  // The unrelated periodic seam lies in the final cell, not this counterexample.
  const a = .015, b = .035;
  const primitive = (x: number) => x <= a ? x : x >= b ? (a + b) / 2 : a + (x - a) - (x - a) ** 2 / (2 * (b - a));
  const plane: SeparableFactor = { value: x => Math.max(0, Math.min(1, (b - x) / (b - a))),
    derivative: x => x > a && x < b ? -1 / (b - a) : 0, integral: (lo, hi) => primitive(hi) - primitive(lo) };
  const field = initializeTensorCSL4([4, 3, 3], [1, 1, 1], [{ scale: 1, factors: [plane, one, one] }]);
  const before = field.data.slice(), amount = tensorAmount(field), report = tensorRangeAdmission(field);
  assert.equal(report.admitted, false);
  assert.ok(report.rejectedCells > 0 && report.lower < -.25);
  near(sampleTensorCSL4(field, [.125, .37, .61])[0], -.25, 1e-13);
  near(amount, .025, 1e-14);
  assert.deepEqual(field.data, before, "range rejection must not silently clamp or alter the current field");
  const local = tensorCellBernstein(field, [0, 0, 0]);
  assert.ok(Math.min(...local) < -1.4);
});

test("an admitted C1 saturated field can lose positivity after translation, requiring candidate rejection", () => {
  // Four exact cubic cells: dry, smooth rise, full, smooth fall. This field is
  // C1 and its initial Bernstein net proves 0<=q<=1 over the whole domain.
  const field = initializeTensorCSL4([4, 2, 2], [4, 2, 2], []);
  for (let z = 0; z < 6; z++) for (let y = 0; y < 6; y++) for (let x = 0; x < 12; x++) {
    const i = Math.floor(x / 3), kind = x % 3;
    field.data[x + 12 * (y + 6 * z)] = y % 3 === 1 || z % 3 === 1 ? 0
      : kind === 0 ? [0, 0, 1, 1][i]! : kind === 1 ? 0 : [0, .5, 1, .5][i]!;
  }
  const original = field.data.slice();
  assert.deepEqual(tensorRangeAdmission(field), { admitted: true, lower: 0, upper: 1, rejectedCells: 0 });
  const candidate = translateTensorCSL4(field, [.5, 0, 0]);
  const range = tensorRangeAdmission(candidate);
  assert.equal(range.admitted, false);
  near(range.lower, -.15625); near(range.upper, 1.15625);
  let minimum = Infinity, maximum = -Infinity;
  for (let i = 0; i < 400; i++) {
    const q = sampleTensorCSL4(candidate, [i / 100, .3, .7])[0];
    minimum = Math.min(minimum, q); maximum = Math.max(maximum, q);
  }
  assert.ok(minimum < -.028 && maximum > 1.028, "this failed bound corresponds to actual overshoot, not only a conservative certificate");
  near(tensorAmount(candidate), tensorAmount(field));
  assert.deepEqual(field.data, original, "forming a rejected candidate must preserve the accepted source exactly");
  const face = tensorFaceIdentity(candidate);
  assert.ok(face.valueError < 1e-13 && face.normalError < 1e-12,
    "continuity and conservation alone do not prevent negative current density");
});

test("resolved smooth nonquadratic 3D orbits retain C1 traces, one-field mass, bounded range and analytic convergence", () => {
  const results: { n: number; qError: number; gradientError: number; milliseconds: number; range: readonly [number, number] }[] = [];
  for (const n of [4, 8, 16]) {
    const start = performance.now();
    let field = wave(n);
    const originalMass = tensorAmount(field), shift: Triple = [1 / (3 * n), 2 / (3 * n), -1 / (3 * n)];
    let lower = Infinity, upper = -Infinity;
    for (let step = 0; step <= 3 * n; step++) {
      const range = tensorRangeAdmission(field);
      assert.ok(range.admitted, `n=${n}, step=${step}: ${JSON.stringify(range)}`);
      lower = Math.min(lower, range.lower); upper = Math.max(upper, range.upper);
      near(tensorAmount(field), originalMass, 3e-12);
      if (step === 0 || step === 3 * n) {
        const face = tensorFaceIdentity(field);
        assert.ok(face.valueError < 2e-13 && face.normalError < 2e-11, JSON.stringify(face));
      }
      if (step !== 3 * n) field = translateTensorCSL4(field, shift);
    }
    let qError = 0, gradientError = 0;
    for (let z = 0; z < 13; z++) for (let y = 0; y < 13; y++) for (let x = 0; x < 13; x++) {
      const p: Triple = [(x + .137) / 13, (y + .271) / 13, (z + .413) / 13], actual = sampleTensorCSL4(field, p), exact = waveExact(p);
      qError = Math.max(qError, Math.abs(actual[0] - exact[0]));
      for (let axis = 1; axis < 4; axis++) gradientError = Math.max(gradientError, Math.abs(actual[axis]! - exact[axis]!));
    }
    results.push({ n, qError, gradientError, milliseconds: performance.now() - start, range: [lower, upper] });
  }
  console.log("tensor CSL4 bounded CPU orbit", JSON.stringify(results));
  for (let i = 0; i < results.length - 1; i++) {
    assert.ok(results[i]!.qError / results[i + 1]!.qError > 16, JSON.stringify(results));
    assert.ok(results[i]!.gradientError / results[i + 1]!.gradientError > 8, JSON.stringify(results));
  }
  assert.ok(results[1]!.qError < 5e-5 && results[1]!.gradientError < 2e-3, JSON.stringify(results));
  assert.ok(results[2]!.qError < 2e-6 && results[2]!.gradientError < 2e-5, JSON.stringify(results));
});
