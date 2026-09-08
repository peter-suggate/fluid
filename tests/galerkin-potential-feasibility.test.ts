import assert from "node:assert/strict";
import test from "node:test";

// Independent counterexample, not an implementation of the proposed solver.
// A finite multiplier would minimize bounded quadratic entropy at fixed spline
// moments. The unique minimizer below is discontinuous despite a C2 source psi.
type Rational = readonly [bigint, bigint];
const gcd = (a: bigint, b: bigint): bigint => b === 0n ? (a < 0n ? -a : a) : gcd(b, a % b);
function rational(n: bigint, d = 1n): Rational {
  assert.notEqual(d, 0n);
  const g = gcd(n, d) * (d < 0n ? -1n : 1n);
  return [n / g, d / g];
}
const add = (a: Rational, b: Rational) => rational(a[0] * b[1] + b[0] * a[1], a[1] * b[1]);
const multiply = (a: Rational, b: Rational) => rational(a[0] * b[0], a[1] * b[1]);
function inner(a: readonly Rational[], b: readonly Rational[]): Rational {
  let result = rational(0n);
  for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++)
    result = add(result, multiply(multiply(a[i]!, b[j]!), rational(1n, BigInt(i + j + 1))));
  return result;
}
const source = [0n, 1n, 0n, 0n, -1n].map(n => rational(n));
const minimum = [1n, 50n, 90n, -140n].map(n => rational(n, 70n));
const close = (a: number, b: number, tolerance = 3e-14) =>
  assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
const clamp = (q: number) => Math.max(0, Math.min(1, q));
const p = (t: number) => (1 + 50 * t + 90 * t * t - 140 * t ** 3) / 70;

function cubic(t: number): number {
  const a = Math.abs(t);
  if (a >= 2) return 0;
  return a >= 1 ? (2 - a) ** 3 / 6 : (4 - 6 * a * a + 3 * a ** 3) / 6;
}
function periodicBasis(t: number, index: number): number {
  return cubic(t - index - 8) + cubic(t - index) + cubic(t - index + 8);
}

// C2 cutoff: one on [0,1], zero off [-.5,1.5]. The exact source is
// psi=.5-(chi*(t-t^4+1)-1). Outside [0,1] its density is identically zero.
function cutoffJet(t: number): readonly [number, number, number] {
  if (t <= -.5 || t >= 1.5) return [0, 0, 0];
  if (t >= 0 && t <= 1) return [1, 0, 0];
  const scale = t < 0 ? 2 : -2, s = t < 0 ? 2 * t + 1 : 3 - 2 * t;
  return [s ** 3 * (10 - 15 * s + 6 * s * s),
    scale * 30 * s * s * (1 - s) ** 2,
    scale * scale * 60 * s * (1 - s) * (1 - 2 * s)];
}
function periodicSourceJet(x: number, amplitude = 1): readonly [number, number, number] {
  const t = ((x + 4) % 8 + 8) % 8 - 4;
  const [c, dc, ddc] = cutoffJet(t), g = t - t ** 4, dg = 1 - 4 * t ** 3, ddg = -12 * t * t;
  return [.5 + amplitude * (1 - c * (g + 1)), amplitude * (-dc * (g + 1) - c * dg),
    amplitude * (-ddc * (g + 1) - 2 * dc * dg - c * ddg)];
}

test("exact P3 moments have a strictly bounded polynomial entropy minimizer with a nonzero endpoint", () => {
  for (let k = 0; k < 4; k++) {
    const monomial = Array.from({ length: k + 1 }, (_, i) => rational(i === k ? 1n : 0n));
    assert.deepEqual(inner(source, monomial), inner(minimum, monomial));
  }
  assert.deepEqual(inner(source, [rational(1n)]), rational(3n, 10n));
  // Cubic Bernstein coefficients certify 0<p<1 on the entire unit cell.
  const bernstein = [rational(1n, 70n), rational(53n, 210n), rational(193n, 210n), rational(1n, 70n)];
  for (const [n, d] of bernstein) assert.ok(n > 0n && n < d);
  for (let i = 0; i <= 100; i++) {
    const t = i / 100, b = bernstein.map(([n, d]) => Number(n) / Number(d));
    close(p(t), b[0]! * (1 - t) ** 3 + 3 * b[1]! * t * (1 - t) ** 2 + 3 * b[2]! * t * t * (1 - t) + b[3]! * t ** 3);
  }
  // Orthogonality gives ||q||²=||p||²+||q-p||² for any feasible q on
  // this cell; bounds do not change the unique minimizer because p is interior.
  assert.deepEqual(add(inner(source, source), multiply(rational(-1n), inner(minimum, minimum))), rational(1n, 44100n));
  close(p(0), 1 / 70); close(p(1), 1 / 70);
});

test("the periodic positive cubic basis forces every other cell dry and spans all four wet-cell moments", () => {
  const active = new Set([7, 0, 1, 2]);
  for (let cell = 0; cell < 8; cell++) {
    for (const fraction of [.1, .4, .9]) {
      const values = Array.from({ length: 8 }, (_, i) => periodicBasis(cell + fraction, i));
      assert.ok(values.every(value => value >= 0));
      close(values.reduce((a, b) => a + b, 0), 1);
      if (cell === 0) for (let i = 0; i < 8; i++) assert.equal(values[i]! > 0, active.has(i));
      else assert.ok(values.some((value, i) => !active.has(i) && value > 0));
    }
  }
  // Restrictions of B7,B0,B1,B2 on (0,1), multiplied by 6.
  // Their power-coefficient determinant is 108, so they span P3.
  const rows = [[1n, -3n, 3n, -1n], [4n, 0n, -6n, 3n], [1n, 3n, 3n, -3n], [0n, 0n, 0n, 1n]];
  const det = rows[0]![0]! * (rows[1]![1]! * rows[2]![2]! - rows[1]![2]! * rows[2]![1]!)
    - rows[0]![1]! * (rows[1]![0]! * rows[2]![2]! - rows[1]![2]! * rows[2]![0]!)
    + rows[0]![2]! * (rows[1]![0]! * rows[2]![1]! - rows[1]![1]! * rows[2]![0]!);
  assert.equal(det, 108n);
  for (let j = 0; j < rows.length; j++) {
    const row = rows[j]!.map(n => rational(n, 6n));
    assert.deepEqual(inner(source, row), inner(minimum, row));
    for (const t of [.1, .4, .9]) close(periodicBasis(t, [7, 0, 1, 2][j]!),
      rows[j]!.reduce((sum, n, k) => sum + Number(n) * t ** k / 6, 0));
  }
});

test("the source is generated by an explicitly periodic C2 latent potential, without a sharp density source", () => {
  const expectedJets = new Map<number, readonly [number, number, number]>([
    [-.5, [1.5, 0, 0]], [0, [.5, -1, 0]], [1, [.5, 3, 12]], [1.5, [1.5, 0, 0]],
  ]);
  for (const [x, expected] of expectedJets) {
    const jet = periodicSourceJet(x);
    for (let k = 0; k < 3; k++) close(jet[k]!, expected[k]!);
    for (const side of [-1, 1]) {
      const errors = [1e-4, 5e-5].map(epsilon => Math.max(...periodicSourceJet(x + side * epsilon).map((v, k) => Math.abs(v - expected[k]!))));
      assert.ok(errors[1]! <= .51 * errors[0]!, `C2 trace at ${x}, side ${side}: ${errors}`);
    }
  }
  for (let i = -200; i <= 200; i++) {
    const x = i / 50, jet = periodicSourceJet(x);
    const expected = x > 0 && x < 1 ? x - x ** 4 : 0;
    close(clamp(.5 - jet[0]), expected);
    periodicSourceJet(x + 8).forEach((value, k) => close(value, jet[k]!, 3e-12));
  }
});

test("no continuous clamped potential can equal the unique periodic moment minimizer", () => {
  // All inactive positive basis moments are exactly zero. Positivity therefore
  // forces q=0 almost everywhere on the seven other open cells. Continuity
  // would force q(0)=q(1)=0, whereas the unique minimizer has both traces 1/70.
  const minimizer = (t: number) => t > 0 && t < 1 ? p(t) : 0;
  close(minimizer(-1e-9), 0); close(minimizer(1 + 1e-9), 0);
  assert.ok(minimizer(1e-9) > 1 / 140 && minimizer(1 - 1e-9) > 1 / 140);
  // Diverging coefficients are not an accepted solution. The standard local
  // saturating direction lambda*(-t)_+^3 is C2 and vanishes on the wet cell,
  // yet every finite lambda leaves the same positive density at the knot.
  for (const lambda of [0, 1, 100, 1e12]) {
    const psiAtKnot = .5 - p(0) + lambda * Math.max(0, -0) ** 3;
    close(clamp(.5 - psiAtKnot), 1 / 70);
  }
});

test("the same obstruction includes a genuine half-density component after bounded amplitude scaling", () => {
  const amplitude = rational(27n, 25n);
  const scaledSource = source.map(c => multiply(c, amplitude));
  const scaledMinimum = minimum.map(c => multiply(c, amplitude));
  for (let k = 0; k < 4; k++) {
    const monomial = Array.from({ length: k + 1 }, (_, i) => rational(i === k ? 1n : 0n));
    assert.deepEqual(inner(scaledSource, monomial), inner(scaledMinimum, monomial));
  }
  // The largest Bernstein coefficient of p stays below one. The source's
  // degree-four coefficients [0,1/4,1/2,3/4,0] also stay below one.
  assert.deepEqual(multiply(rational(193n, 210n), amplitude), rational(5211n, 5250n));
  assert.ok(5211n < 5250n);
  assert.deepEqual(multiply(rational(3n, 4n), amplitude), rational(81n, 100n));
  const a = 27 / 25;
  assert.ok(a * (5 / 8 - (5 / 8) ** 4) > .5);
  assert.ok(a * p(5 / 8) > .5);
  close(clamp(.5 - periodicSourceJet(0, a)[0]), 0);
  close(clamp(.5 - periodicSourceJet(1, a)[0]), 0);
  assert.ok(clamp(.5 - periodicSourceJet(5 / 8, a)[0]) > .5);
  // Thus the continuous source crosses q=.5 on both sides of this point,
  // while the unique moment minimizer still jumps at the dry-cell boundary.
  close(a * p(0), 27 / 1750);
});
