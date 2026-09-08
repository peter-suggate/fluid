import assert from "node:assert/strict";
import test from "node:test";
import {
  advanceWeakDensityLine, averageClampedSegment, integrateWeakDensityLine, sampleWeakDensityLine,
  type WeakDensityLine,
} from "../tools/implicit-density/weak-segment-density-line-oracle";

const TAU = 2 * Math.PI;
const close = (a: number, b: number, tolerance: number) =>
  assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}; tolerance ${tolerance}`);
const wave = (n: number, amplitude: number): WeakDensityLine => {
  const frequency = TAU / n, interpolationFilter = 2 / 3 + Math.cos(frequency) / 3;
  return { width: .05, length: 1,
    coefficients: Float64Array.from({ length: n }, (_, i) => amplitude * Math.sin(frequency * i + .13) / interpolationFilter) };
};

// Independent global cardinal-basis evaluation, rather than the candidate's
// cell-local power-polynomial evaluation. These tests do not use its integrator
// or root isolation for physical amount, geometry, or the translated reference.
function cardinal(x: number, order = 0): number {
  const a = Math.abs(x);
  if (a >= 2) return 0;
  if (order === 0) return a < 1 ? 2 / 3 - a * a + a ** 3 / 2 : (2 - a) ** 3 / 6;
  if (order === 1) return Math.sign(x) * (a < 1 ? -2 * a + 1.5 * a * a : -((2 - a) ** 2) / 2);
  return a < 1 ? -2 + 3 * a : 2 - a;
}
function referencePsi(line: WeakDensityLine, x: number, order = 0): number {
  const n = line.coefficients.length, h = line.length / n, t = ((x / h) % n + n) % n;
  let result = 0;
  for (let i = 0; i < n; i++) for (const period of [-n, 0, n])
    result += line.coefficients[i]! * cardinal(t - i + period, order) / h ** order;
  return result;
}
function referenceRoots(line: WeakDensityLine, threshold: number): number[] {
  const roots: number[] = [], intervals = line.coefficients.length * 32;
  for (let i = 0; i < intervals; i++) {
    let lower = i / intervals, upper = (i + 1) / intervals;
    let a = referencePsi(line, lower) - threshold, b = referencePsi(line, upper) - threshold;
    if (a === 0) roots.push(lower);
    if (!(a * b < 0)) continue;
    for (let iteration = 0; iteration < 48; iteration++) {
      const middle = (lower + upper) / 2, value = referencePsi(line, middle) - threshold;
      if ((value < 0) === (a < 0)) { lower = middle; a = value; } else { upper = middle; b = value; }
    }
    roots.push((lower + upper) / 2);
  }
  return roots;
}
function referenceIntegral(line: WeakDensityLine): { amount: number; squaredAmount: number } {
  // Test fixtures have four simple saturation roots. Independently bracket them
  // on 32 subintervals/cell, then integrate q (cubic) and q² (degree six) with
  // Gauss4 on every knot/root piece. This is not the candidate's Gauss4/8 rule.
  const cuts = [...Array.from({ length: line.coefficients.length + 1 }, (_, i) => i / line.coefficients.length),
    ...referenceRoots(line, -line.width / 2), ...referenceRoots(line, line.width / 2)].sort((a, b) => a - b);
  const x0 = Math.sqrt((3 - 2 * Math.sqrt(6 / 5)) / 7), x1 = Math.sqrt((3 + 2 * Math.sqrt(6 / 5)) / 7);
  const w0 = (18 + Math.sqrt(30)) / 36, w1 = (18 - Math.sqrt(30)) / 36;
  let amount = 0, squaredAmount = 0;
  for (let i = 0; i + 1 < cuts.length; i++) for (const [node, weight] of [[-x1, w1], [-x0, w0], [x0, w0], [x1, w1]]) {
    const half = (cuts[i + 1]! - cuts[i]!) / 2, middle = (cuts[i + 1]! + cuts[i]!) / 2;
    const q = Math.max(0, Math.min(1, .5 - referencePsi(line, middle + half * node!) / line.width));
    amount += half * weight! * q; squaredAmount += half * weight! * q * q;
  }
  return { amount, squaredAmount };
}

function trajectory(n: number, amplitude: number, time: number, steps: number) {
  const original = wave(n, amplitude), originalBytes = original.coefficients.slice();
  const initial = referenceIntegral(original);
  let current = original, maxResidual = 0, maxCondition = 0, maxIterations = 0, quadratureEvaluations = 0;
  for (let step = 0; step < steps; step++) {
    const before = current.coefficients.slice();
    const result = advanceWeakDensityLine(current, 1, time / steps);
    assert.deepEqual(current.coefficients, before, "an accepted step leaves the source generation unchanged");
    current = result.line;
    maxResidual = Math.max(maxResidual, result.receipt.residual);
    maxCondition = Math.max(maxCondition, result.receipt.final.gramCondition);
    maxIterations = Math.max(maxIterations, result.receipt.iterations);
    quadratureEvaluations += result.receipt.totalQuadratureEvaluations;
    assert.ok(result.receipt.final.minimumCholeskyPivot > 0);
  }
  assert.deepEqual(original.coefficients, originalBytes);
  const final = referenceIntegral(current);
  let psiError = 0, derivativeError = 0, analyticError = 0, densityError = 0;
  for (let i = 0; i < 512; i++) {
    const x = (i + .37) / 512, actual = referencePsi(current, x), expected = referencePsi(original, x - time);
    psiError = Math.max(psiError, Math.abs(actual - expected));
    derivativeError = Math.max(derivativeError, Math.abs(referencePsi(current, x, 1) - referencePsi(original, x - time, 1)));
    analyticError = Math.max(analyticError, Math.abs(actual - amplitude * Math.sin(TAU * (x - time) + .13)));
    densityError = Math.max(densityError, Math.abs(Math.max(0, Math.min(1, .5 - actual / .05)) - Math.max(0, Math.min(1, .5 - expected / .05))));
  }
  const actualRoots = referenceRoots(current, 0), expectedRoots = referenceRoots(original, 0).map(root => (root + time) % 1);
  assert.equal(actualRoots.length, 2, "half-density component crossings remain complete");
  let zeroError = 0;
  for (const root of expectedRoots) zeroError = Math.max(zeroError,
    Math.min(...actualRoots.map(actual => Math.min(Math.abs(actual - root), 1 - Math.abs(actual - root)))));
  return { original, current, metrics: { n, amplitude, steps, psiError, derivativeError, analyticError, densityError, zeroError,
    physicalAmountChange: final.amount - initial.amount, squaredAmountChange: final.squaredAmount - initial.squaredAmount,
    maxResidual, maxCondition, maxIterations, quadratureEvaluations } };
}

test("the exact coefficient-segment clamp average and endpoint derivative include all saturation crossings", () => {
  for (const [a, b, expected, firstMoment] of [
    [-2, -1, 1, 0], [1, 2, 0, 0], [-.25, .25, .5, .5],
    [-1, 1, .5, .25], [1, -1, .5, .25], [.75, 0, 1 / 6, 4 / 9], [0, .75, 1 / 6, 2 / 9],
  ]) {
    const result = averageClampedSegment(a!, b!, 1);
    close(result.average, expected!, 2e-15); close(result.rampFirstMoment, firstMoment!, 2e-15);
    const epsilon = 1e-6;
    const derivative = (averageClampedSegment(a!, b! + epsilon, 1).average - averageClampedSegment(a!, b! - epsilon, 1).average) / (2 * epsilon);
    close(derivative, -firstMoment!, 1e-9);
  }
});

test("current weakly saturated field integrals and C2 traces agree with independent physical evaluation", () => {
  for (const n of [8, 16]) {
    const line = wave(n, .026), current = integrateWeakDensityLine(line), physical = referenceIntegral(line);
    assert.ok(current.minimumCholeskyPivot > 0 && current.gramCondition < 200);
    close(current.amount, physical.amount, 3e-14); close(current.squaredAmount, physical.squaredAmount, 3e-14);
    assert.equal(referenceRoots(line, line.width / 2).length, 2);
    assert.equal(referenceRoots(line, -line.width / 2).length, 2);
    for (let i = 0; i <= n; i++) for (const offset of [-1e-7, 0, 1e-7]) {
      const x = i / n + offset, actual = sampleWeakDensityLine(line, x);
      close(actual.psi, referencePsi(line, x), 2e-16);
      close(actual.derivative, referencePsi(line, x, 1), 3e-15);
    }
    // Continuous second derivatives at every knot, with shrinking error to the
    // common cardinal-basis trace; the potential's clamp need only be C0.
    for (let i = 0; i < n; i++) {
      const x = i / n;
      for (const side of [-1, 1]) {
        const coarse = Math.abs(referencePsi(line, x + side * 1e-5, 2) - referencePsi(line, x, 2));
        const fine = Math.abs(referencePsi(line, x + side * 5e-6, 2) - referencePsi(line, x, 2));
        assert.ok(fine <= .501 * coarse + 1e-13);
      }
    }
  }
});

test("unsaturated current-field translation has second-order time convergence and reversible amount conservation", () => {
  const runs = [4, 8, 16].map(steps => trajectory(16, .015, .125, steps));
  for (let i = 1; i < runs.length; i++) assert.ok(runs[i]!.metrics.psiError < .27 * runs[i - 1]!.metrics.psiError);
  for (const run of runs) {
    assert.ok(Math.abs(run.metrics.physicalAmountChange) < 2e-13);
    assert.ok(Math.abs(run.metrics.squaredAmountChange) < 2e-13);
  }
  const forward = advanceWeakDensityLine(wave(16, .015), 1, 1 / 64);
  const reverse = advanceWeakDensityLine(forward.line, 1, -1 / 64);
  reverse.line.coefficients.forEach((coefficient, i) => close(coefficient, wave(16, .015).coefficients[i]!, 2e-15));
  console.log("weak-segment unsaturated time convergence", JSON.stringify(runs.map(run => run.metrics)));
});

test("weakly saturated translation refines in space, preserves physical amount, and reports finite-step squared-density drift", () => {
  const runs = [8, 16].map(n => trajectory(n, .026, .125, 128));
  assert.ok(runs[1]!.metrics.analyticError < .15 * runs[0]!.metrics.analyticError);
  assert.ok(runs[1]!.metrics.zeroError < runs[0]!.metrics.zeroError);
  for (const run of runs) {
    assert.ok(Math.abs(run.metrics.physicalAmountChange) < 2e-11);
    assert.ok(run.metrics.maxCondition < 1e4);
    assert.ok(run.metrics.densityError < .001);
    assert.ok(run.metrics.maxResidual <= 1e-12);
  }
  const large = advanceWeakDensityLine(wave(16, .026), 1, 1 / 64);
  const small = advanceWeakDensityLine(wave(16, .026), 1, 1 / 128);
  assert.ok(Math.abs(large.receipt.squaredAmountChange) > 1e-9, "finite-step entropy conservation must not be claimed");
  assert.ok(Math.abs(small.receipt.squaredAmountChange) < Math.abs(large.receipt.squaredAmountChange));
  const reverse = advanceWeakDensityLine(small.line, 1, -1 / 128);
  reverse.line.coefficients.forEach((coefficient, i) => close(coefficient, wave(16, .026).coefficients[i]!, 3e-11));
  // Temporal refinement at fixed N, separate from the actual translated-field
  // errors reported above. The 128-step trajectory resolves the same weak PDE;
  // it is not substituted for the independent geometric or physical oracle.
  const temporalErrors = [8, 16, 32].map(steps => {
    const run = trajectory(16, .026, .125, steps);
    return Math.max(...run.current.coefficients.map((coefficient, i) => Math.abs(coefficient - runs[1]!.current.coefficients[i]!)));
  });
  for (let i = 1; i < temporalErrors.length; i++) assert.ok(temporalErrors[i]! < .3 * temporalErrors[i - 1]!);
  console.log("weak-segment saturated space convergence", JSON.stringify(runs.map(run => run.metrics)));
  console.log("weak-segment saturated temporal errors", JSON.stringify(temporalErrors));
});

test("an asymmetric clipped field conserves non-half physical amount and scales consistently with physical length", () => {
  const seed = wave(16, .026), original = { ...seed, coefficients: seed.coefficients.map(value => value + .0007) };
  const initial = referenceIntegral(original);
  assert.ok(Math.abs(initial.amount - .5) > .005, "mass gate must not follow from half-period antisymmetry");
  let current: WeakDensityLine = original;
  let maximumPhysicalDrift = 0;
  for (let i = 0; i < 32; i++) {
    const next = advanceWeakDensityLine(current, 1, 1 / 256);
    current = next.line;
    const physical = referenceIntegral(current);
    maximumPhysicalDrift = Math.max(maximumPhysicalDrift, Math.abs(physical.amount - initial.amount));
    close(next.receipt.final.amount, physical.amount, 5e-14);
  }
  assert.ok(maximumPhysicalDrift < 2e-11);
  const step = advanceWeakDensityLine(original, 1, 1 / 256);
  const scaled = advanceWeakDensityLine({ ...original, length: 2 }, 2, 1 / 256);
  scaled.line.coefficients.forEach((coefficient, i) => close(coefficient, step.line.coefficients[i]!, 2e-15));
  close(scaled.receipt.final.amount, 2 * step.receipt.final.amount, 2e-14);
  console.log("weak-segment asymmetric physical mass", JSON.stringify({ initialAmount: initial.amount, maximumPhysicalDrift }));
});

test("unsupported threshold plateaus, conditioning, integration and nonlinear solves reject without source mutation", () => {
  const line = wave(16, .026), bytes = line.coefficients.slice();
  for (const options of [{ maxNewtonIterations: 0 }, { maxGramCondition: 1 }, { maxLinearCondition: 1 },
    { maxQuadratureDepth: 0 }, { maxQuadratureEvaluations: 1 }]) {
    assert.throws(() => advanceWeakDensityLine(line, 1, 1 / 64, options), /rejected|cap/);
    assert.deepEqual(line.coefficients, bytes);
  }
  const plateau = { ...line, coefficients: new Float64Array(16).fill(.025) };
  plateau.coefficients[0] = .035; // psi=.025+.01*B0, exactly dry with threshold plateaus
  const plateauBytes = plateau.coefficients.slice();
  assert.throws(() => advanceWeakDensityLine(plateau, 1, 1 / 64), /threshold plateau/);
  assert.deepEqual(plateau.coefficients, plateauBytes);
  assert.throws(() => advanceWeakDensityLine(line, 1, 1), /half a cell/);
  assert.throws(() => advanceWeakDensityLine(line, NaN, 1 / 64), /finite uniform/);
  const identity = advanceWeakDensityLine(line, 0, 0, { maxNewtonIterations: 0 });
  assert.deepEqual(identity.line.coefficients, bytes);
  assert.notEqual(identity.line.coefficients.buffer, line.coefficients.buffer);
});
