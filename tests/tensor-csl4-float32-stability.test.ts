import assert from "node:assert/strict";
import test from "node:test";
import { hermiteCell, quarticDerivative, quarticValue } from "../tools/implicit-density/conservative-hermite-oracle";
import { initializeTensorCSL4, translateTensorCSL4, type TensorCSL4Field, type Triple } from "../tools/implicit-density/tensor-csl4-oracle";

// Arithmetic audit, not a CPU runtime or exact model of Metal's permitted FMA
// contraction. Every primitive operation rounds to f32. Actual Dawn agreement
// remains a separate gate with unchanged tolerances.
const f = Math.fround;
const add = (...xs: number[]) => xs.reduce((a, b) => f(a + b));
const mul = (...xs: number[]) => xs.reduce((a, b) => f(a * b));
const sub = (a: number, b: number) => f(a - b);
const div = (a: number, b: number) => f(a / b);
const POWER = [[1, 0, -18, 32, -15], [0, 1, -4.5, 6, -2.5], [0, 0, -12, 28, -15],
  [0, 0, 1.5, -4, 2.5], [0, 0, 30, -60, 30]];
function oldBasis(t: number, derivative: boolean): number[] {
  if (t === 0 || t === 1) return Array.from({ length: 5 }, (_, i) => Number(i === (t === 0 ? 0 : 2) + Number(derivative)));
  return POWER.map(c => derivative
    ? add(c[1]!, mul(t, add(mul(2, c[2]!), mul(t, add(mul(3, c[3]!), mul(4, t, c[4]!))))))
    : add(c[0]!, mul(t, add(c[1]!, mul(t, add(c[2]!, mul(t, add(c[3]!, mul(t, c[4]!)))))))));
}
function oldPrefix(t: number): number[] {
  if (t === 0 || t === 1) return [0, 0, 0, 0, t];
  return POWER.map(c => mul(t, add(c[0]!, mul(t, add(mul(.5, c[1]!), mul(t,
    add(div(c[2]!, 3), mul(t, add(mul(.25, c[3]!), div(mul(t, c[4]!), 5))))))))));
}
const dot = (v: readonly number[], w: readonly number[]) => v.reduce((sum, value, i) => add(sum, mul(value, w[i]!)), 0);
const residual = (v: readonly number[]) => sub(sub(v[4]!, mul(.5, add(v[0]!, v[2]!))), div(sub(v[1]!, v[3]!), 12));
function stablePointLeft(v: readonly number[], t: number, derivative: boolean): number {
  if (t === 0 || t === 1) return v[(t === 0 ? 0 : 2) + Number(derivative)]!;
  const u = sub(1, t), difference = sub(v[2]!, v[0]!), delta = residual(v);
  if (derivative) return add(mul(difference, 6, t, u), mul(v[1]!, u, sub(1, mul(3, t))),
    mul(v[3]!, t, sub(mul(3, t), 2)), mul(delta, 60, t, u, sub(1, mul(2, t))));
  return add(v[0]!, mul(difference, t, t, sub(3, mul(2, t))), mul(v[1]!, t, u, u),
    -mul(v[3]!, t, t, u), mul(delta, 30, t, t, u, u));
}
function stablePrefixLeft(v: readonly number[], t: number): number {
  if (t === 0) return 0; if (t === 1) return v[4]!;
  const t2 = mul(t, t), t3 = mul(t2, t);
  return add(mul(v[0]!, t), mul(sub(v[2]!, v[0]!), t3, sub(1, mul(.5, t))),
    mul(v[1]!, t2, add(.5, mul(t, add(f(-2 / 3), mul(.25, t))))),
    mul(v[3]!, t3, add(f(-1 / 3), mul(.25, t))), mul(residual(v), t3, add(10, mul(t, add(-15, mul(6, t))))));
}
const reflected = (v: readonly number[]) => [v[2]!, -v[3]!, v[0]!, -v[1]!, v[4]!];
function stablePoint(v: readonly number[], t: number, derivative: boolean): number {
  if (t <= .5) return stablePointLeft(v, t, derivative);
  const value = stablePointLeft(reflected(v), sub(1, t), derivative); return derivative ? -value : value;
}
function stablePrefix(v: readonly number[], t: number): number {
  return t <= .5 ? stablePrefixLeft(v, t) : sub(v[4]!, stablePrefixLeft(reflected(v), sub(1, t)));
}
function stableMean(left: readonly number[], right: readonly number[], t: number): number {
  const difference = right.map((value, i) => sub(value, left[i]!));
  return t <= .5 ? add(left[4]!, stablePrefixLeft(difference, t))
    : add(right[4]!, -stablePrefixLeft(reflected(difference), sub(1, t)));
}
function translateFloat32(field: TensorCSL4Field, displacement: Triple, stable: boolean): TensorCSL4Field {
  const sizes = field.dimensions.map(n => 3 * n), strides = [1, sizes[0]!, sizes[0]! * sizes[1]!];
  let source = Float32Array.from(field.data);
  for (let axis = 0; axis < 3; axis++) {
    const n = field.dimensions[axis]!, raw = div(f(displacement[axis]!), f(field.lengths[axis]! / n));
    const shift = sub(raw, mul(Math.floor(div(raw, n)), n)), whole = Math.floor(shift), fraction = sub(shift, whole), t = sub(1, fraction);
    const stride = strides[axis]!, block = stride * sizes[axis]!, next = new Float32Array(source.length);
    const wrap = (i: number) => ((i % n) + n) % n;
    for (let base = 0; base < source.length; base += block) for (let offset = 0; offset < stride; offset++) {
      const moments = (i: number) => [3 * wrap(i), 3 * wrap(i) + 1, 3 * wrap(i + 1), 3 * wrap(i + 1) + 1, 3 * wrap(i) + 2]
        .map(slot => source[base + offset + slot * stride]!);
      for (let slot = 0; slot < sizes[axis]!; slot++) {
        const cell = Math.floor(slot / 3), kind = slot % 3, at = base + offset + slot * stride;
        if (fraction === 0) { next[at] = source[base + offset + (3 * wrap(cell - whole) + kind) * stride]!; continue; }
        const donor = cell - whole - 1, left = moments(donor);
        if (kind !== 2) next[at] = stable ? stablePoint(left, t, kind === 1) : dot(left, oldBasis(t, kind === 1));
        else {
          const right = moments(donor + 1);
          next[at] = stable ? stableMean(left, right, t)
            : add(dot(left, oldPrefix(t).map((weight, i) => sub(Number(i === 4), weight))), dot(right, oldPrefix(t)));
        }
      }
    }
    source = next;
  }
  return { ...field, data: Float64Array.from(source) };
}
const difference = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from({ length: a.length }, (_, i) => Math.abs(a[i]! - b[i]!)).reduce((x, y) => Math.max(x, y), 0);
const one = { value: () => 1, derivative: () => 0, integral: (a: number, b: number) => b - a };

test("factored value, derivative and prefix measure the same independent quartic", () => {
  for (const raw of [[.37, 0, .37, 0, .37], [.2, .11, .8, -.07, .51], [.9, -.3, .1, .2, .4]]) {
    const v = raw.map(f), p = hermiteCell({ length: 2, value: Float64Array.of(v[0]!, v[2]!),
      derivative: Float64Array.of(v[1]!, v[3]!), amount: Float64Array.of(v[4]!, .5) }, 0);
    for (const rawT of [0, .13, 1 / 3, .5, .87, 1]) {
      const t = f(rawT), prefix = p.reduce((sum, coefficient, k) => sum + coefficient * t ** (k + 1) / (k + 1), 0);
      assert.ok(Math.abs(stablePoint(v, t, false) - quarticValue(p, t)) < 5e-7);
      assert.ok(Math.abs(stablePoint(v, t, true) - quarticDerivative(p, t)) < 2e-6);
      assert.ok(Math.abs(stablePrefix(v, t) - prefix) < 5e-7);
    }
  }
});

test("factored float32 tensor transport preserves arbitrary constant values and zero derivatives exactly", () => {
  const initial = initializeTensorCSL4([4, 4, 4], [1, 1, 1], [{ scale: .37, factors: [one, one, one] }]);
  initial.data.set(Float32Array.from(initial.data)); let old = initial, stable = initial;
  for (let step = 0; step < 12; step++) {
    old = translateFloat32(old, [1 / 12, 1 / 6, -1 / 12], false);
    stable = translateFloat32(stable, [1 / 12, 1 / 6, -1 / 12], true);
  }
  assert.deepEqual(stable.data, initial.data);
  const oldError = difference(old.data, initial.data);
  assert.ok(oldError > 1e-6, `expanded cardinal constant drift ${oldError}`);
});

test("factored current-DOF float32 orbit reduces arithmetic error without changing the target polynomial", () => {
  const tau = 2 * Math.PI;
  const sin = { value: (x: number) => Math.sin(tau * x), derivative: (x: number) => tau * Math.cos(tau * x),
    integral: (a: number, b: number) => (Math.cos(tau * a) - Math.cos(tau * b)) / tau };
  const cos = { value: (x: number) => Math.cos(tau * x), derivative: (x: number) => -tau * Math.sin(tau * x),
    integral: (a: number, b: number) => (Math.sin(tau * b) - Math.sin(tau * a)) / tau };
  const results = [];
  for (const n of [4, 8]) {
    const initial = initializeTensorCSL4([n, n, n], [1, 1, 1], [{ scale: .5, factors: [one, one, one] },
      { scale: .1, factors: [sin, sin, sin] }, { scale: .07, factors: [cos, cos, cos] }]);
    initial.data.set(Float32Array.from(initial.data)); let exact = initial, old = initial, stable = initial;
    const shift: Triple = [1 / (3 * n), 2 / (3 * n), -1 / (3 * n)];
    for (let step = 0; step < 3 * n; step++) {
      exact = translateTensorCSL4(exact, shift);
      old = translateFloat32(old, shift, false);
      stable = translateFloat32(stable, shift, true);
    }
    results.push({ n, oldError: difference(old.data, exact.data), stableError: difference(stable.data, exact.data) });
  }
  console.log(JSON.stringify({ audit: "strict float32 tensor orbit arithmetic", results, exactModel: "current float64 tensor" }));
  for (const result of results) assert.ok(result.stableError < 3e-5 && result.stableError < result.oldError, JSON.stringify(results));
});
