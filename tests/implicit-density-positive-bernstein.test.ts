import assert from "node:assert/strict";
import test from "node:test";
import { bernsteinCellMeans, compileBernsteinSupport, evaluateBernsteinCell,
  fitPositiveBernsteinMeans, integrateBernstein, positiveBernsteinField,
  type Point3 } from "../tools/implicit-density/positive-bernstein";

const near = (a: number, b: number, tolerance = 2e-13) => assert.ok(Math.abs(a - b) <= tolerance,
  `${a} != ${b}, error ${Math.abs(a - b)}`);
const boxes = [{ lower: [0, 0, 0] as const, width: 1 }, { lower: [1, 0, 0] as const, width: 1 }];
const polynomial = ([x, y, z]: Point3) => 0.6 + 0.1 * x + 0.04 * y + 0.01 * z + 0.07 * x * x - 0.03 * y * y + 0.015 * x * y;
function quadraticField() {
  const support = compileBernsteinSupport(boxes);
  const controls = support.positions.map(([x, y, z]) => polynomial([x, y, z])
    - (Number.isInteger(x) ? 0 : 0.07 / 4) + (Number.isInteger(y) ? 0 : 0.03 / 4));
  return positiveBernsteinField(support, controls);
}
// Independent two-point Gauss quadrature is exact for each tensor quadratic.
function quadrature(f: (p: Point3) => number, lower: Point3, upper: Point3) {
  let sum = 0;
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
    const p = [x, y, z].map((s, a) => (lower[a] + upper[a]) / 2
      + s * (upper[a] - lower[a]) / (2 * Math.sqrt(3))) as unknown as Point3;
    sum += f(p);
  }
  return sum * upper.reduce((v, u, a) => v * (u - lower[a]), 1) / 8;
}

test("shared quadratic basis exactly retains smooth curvature and mixed terms", () => {
  const field = quadraticField();
  assert.equal(field.controls.length, 45, "two cells share their nine face controls");
  for (let cell = 0; cell < 2; cell++) for (let i = 0; i < 23; i++) {
    const p: Point3 = [cell + (i + 0.3) / 24, ((i * 7) % 23 + 0.2) / 24, ((i * 11) % 23 + 0.4) / 24];
    near(evaluateBernsteinCell(field, cell, p), polynomial(p));
  }
  near(integrateBernstein(field, [0.13, 0.27, 0.09], [1.81, 0.92, 0.87]),
    quadrature(polynomial, [0.13, 0.27, 0.09], [1.81, 0.92, 0.87]));
});

test("arbitrary nonnegative target means preserve all shared face traces and positivity", () => {
  const field = quadraticField();
  for (const target of [[0, 1.2], [0.003, 0.017], [1.8, 0.12], [0, 0]]) {
    const fitted = fitPositiveBernsteinMeans(field, target).field;
    bernsteinCellMeans(fitted).forEach((v, i) => near(v, target[i]));
    assert.ok(fitted.controls.every(v => v >= 0));
    for (let i = 0; i < 31; i++) {
      const p: Point3 = [1, (i + 0.1) / 32, ((i * 7) % 31 + 0.5) / 32];
      near(evaluateBernsteinCell(fitted, 0, p), evaluateBernsteinCell(fitted, 1, p));
      for (let cell = 0; cell < 2; cell++) assert.ok(evaluateBernsteinCell(fitted, cell,
        [cell + (i + 0.3) / 32, p[1], p[2]]) >= 0);
    }
    for (let cell = 0; cell < 2; cell++) near(quadrature(p => evaluateBernsteinCell(fitted, cell, p),
      [cell, 0, 0], [cell + 1, 1, 1]), target[cell]);
  }
});

test("100 independently changed query partitions preserve canonical coefficients and total amount", () => {
  const field = quadraticField(), before = field.controls.slice();
  const total = quadrature(polynomial, [0, 0, 0], [2, 1, 1]);
  for (let cycle = 0; cycle < 100; cycle++) {
    const cut = 0.01 + 1.98 * (cycle + 0.5) / 100;
    const a = integrateBernstein(field, [0, 0, 0], [cut, 1, 1]);
    const b = integrateBernstein(field, [cut, 0, 0], [2, 1, 1]);
    near(a + b, total);
    near(a, quadrature(polynomial, [0, 0, 0], [cut, 1, 1]));
  }
  assert.deepEqual(field.controls, before);
});

test("unchanged means retain the existing curved field up to subtraction roundoff", () => {
  const initial = quadraticField(), original = initial.controls.slice();
  const fit = fitPositiveBernsteinMeans(initial, bernsteinCellMeans(initial));
  near(fit.maximumTraceChange, 0);
  fit.field.controls.forEach((v, i) => near(v, original[i]));
});

test("shared traces allow an intentional crease at a retained support boundary", () => {
  const support = compileBernsteinSupport(boxes);
  const field = positiveBernsteinField(support, support.positions.map(([x, y]) => 0.5 + 0.2 * Math.abs(x - 1) - 0.1 * y));
  const epsilon = 1e-5;
  const crease: Point3 = [1, 0.25, 0.4];
  const q = evaluateBernsteinCell(field, 0, crease);
  const left = (q - evaluateBernsteinCell(field, 0, [1 - epsilon, 0.25, 0.4])) / epsilon;
  const right = (evaluateBernsteinCell(field, 1, [1 + epsilon, 0.25, 0.4]) - q) / epsilon;
  near(left, -0.2, 1e-10); near(right, 0.2, 1e-10);
});

test("feasibility correction exposes its geometry risk: a private bubble can create an isolated component", () => {
  const support = compileBernsteinSupport([{ lower: [0, 0, 0], width: 1 }]);
  const zero = positiveBernsteinField(support, new Float64Array(27));
  const fit = fitPositiveBernsteinMeans(zero, [0.3]);
  near(bernsteinCellMeans(fit.field)[0], 0.3);
  near(evaluateBernsteinCell(fit.field, 0, [0.5, 0.5, 0.5]), 1.0125);
  near(evaluateBernsteinCell(fit.field, 0, [0, 0.5, 0.5]), 0);
  assert.ok(fit.maximumInteriorChange > 8);
});

test("nonconforming support is rejected rather than silently introducing hanging-face seams", () => {
  assert.throws(() => compileBernsteinSupport([...boxes, { lower: [2, 0, 0], width: 2 }]));
});

test("field/support snapshots own immutable copies and distinguish density evolution generations", () => {
  const support = compileBernsteinSupport(boxes, 4);
  const supplied = new Float64Array(support.positions.length).fill(0.5);
  const field = positiveBernsteinField(support, supplied, 8);
  supplied.fill(1);
  assert.equal(field.controls[0], 0.5);
  assert.ok(Object.isFrozen(field) && Object.isFrozen(field.controls));
  assert.ok(Object.isFrozen(support) && Object.isFrozen(support.cellControls[0]));
  const next = fitPositiveBernsteinMeans(field, [0.4, 0.6]).field;
  assert.equal(next.generation, 9); assert.equal(next.support, support);
  assert.equal(field.generation, 8); assert.equal(support.generation, 4);
  assert.throws(() => fitPositiveBernsteinMeans(field, [0.5, 0.5], 8));
});

test("de Casteljau support refinement retains smooth q and its integrals exactly", async () => {
  const { refinePositiveBernsteinField } = await import("../tools/implicit-density/positive-bernstein");
  const field = quadraticField(), refined = refinePositiveBernsteinField(field);
  assert.equal(refined.support.generation, field.support.generation + 1);
  assert.equal(refined.generation, field.generation);
  assert.equal(refined.support.boxes.length, 16);
  for (let cell = 0; cell < refined.support.boxes.length; cell++) {
    const box = refined.support.boxes[cell];
    const p = box.lower.map(v => v + 0.173) as unknown as Point3;
    near(evaluateBernsteinCell(refined, cell, p), polynomial(p));
  }
  near(integrateBernstein(refined, [0.03, 0.21, 0.17], [1.93, 0.82, 0.99]),
    integrateBernstein(field, [0.03, 0.21, 0.17], [1.93, 0.82, 0.99]));
});

test("thin endpoint intervals retain positive small Bernstein moments without cancellation", () => {
  const support = compileBernsteinSupport([{ lower: [0, 0, 0], width: 1 }]);
  const controls = support.positions.map(([x]) => x === 0 ? 1 : 0);
  const field = positiveBernsteinField(support, controls);
  const lower = 1 - 1e-8, span = 1 - lower;
  const expected = span ** 3 / 3;
  const actual = integrateBernstein(field, [lower, 0, 0], [1, 1, 1]);
  assert.ok(actual > 0);
  assert.ok(Math.abs(actual / expected - 1) < 1e-7);
});
