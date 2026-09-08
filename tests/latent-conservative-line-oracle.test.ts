import assert from "node:assert/strict";
import test from "node:test";
import { definedDensity, initializeLatentLine, integrateDefinedDensity, integrateLatentLine, latentCell,
  polynomialRoots, polynomialValue, sampleLatentLine, solveLatentCell, transportLatentLine, type LatentLine,
} from "../tools/implicit-density/latent-conservative-line-oracle";

const near = (a: number, b: number, tolerance = 2e-12) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}, error ${Math.abs(a - b)}`);
const total = (line: LatentLine) => line.amount.reduce((a, b) => a + b, 0);
/** Independent analytic branch integration: threshold crossings and primitive
 * are authored explicitly in each fixture, never production root isolation. */
function clippedIntegral(psi: (x: number) => number, primitive: (x: number) => number,
  crossings: readonly number[], width: number, lower: number, upper: number): number {
  const cuts = [lower, ...crossings.filter(x => x > lower && x < upper), upper].sort((a, b) => a - b);
  let result = 0;
  for (let i = 0; i + 1 < cuts.length; i++) {
    const a = cuts[i]!, b = cuts[i + 1]!, value = psi((a + b) / 2);
    if (value <= -width / 2) result += b - a;
    else if (value < width / 2) result += .5 * (b - a) - (primitive(b) - primitive(a)) / width;
  }
  return result;
}
function roots(line: LatentLine): number[] {
  return Array.from({ length: line.bubble.length }, (_, cell) => polynomialRoots(latentCell(line, cell))
    .map(t => line.lower + (cell + t) * line.h)).flat().sort((a, b) => a - b)
    .filter((x, i, values) => i === 0 || x - values[i - 1]! > 1e-10);
}
function assertC1(line: LatentLine): void {
  for (let i = 0; i + 1 < line.bubble.length; i++) {
    const a = latentCell(line, i), b = latentCell(line, i + 1);
    near(polynomialValue(a, 1), b[0]!, 5e-12);
    near(a.slice(1).reduce((sum, c, k) => sum + (k + 1) * c, 0) / line.h, b[1]! / line.h, 5e-11);
  }
}
const width = .05, tau = 2 * Math.PI, amplitude = .08;
function sineIntegral(lower: number, upper: number): number {
  const psi = (x: number) => amplitude * Math.sin(tau * x), primitive = (x: number) => -amplitude * Math.cos(tau * x) / tau;
  const angle = Math.asin(width / (2 * amplitude)) / tau, crossings: number[] = [];
  for (let period = Math.floor(lower) - 1; period <= Math.ceil(upper) + 1; period++) {
    for (const root of [angle, .5 - angle, .5 + angle, 1 - angle]) crossings.push(period + root);
  }
  return clippedIntegral(psi, primitive, crossings, width, lower, upper);
}
function saturatedSine(n: number): LatentLine {
  return initializeLatentLine(n, 0, 1, width, x => amplitude * Math.sin(tau * x),
    x => amplitude * tau * Math.cos(tau * x), sineIntegral, true);
}

test("defined clamped quartic integral resolves both saturation branches and tangent roots", () => {
  // psi=(t-.5)^2/2-.03; its q=0/1 roots are explicit and unrelated to the
  // recursive polynomial-root implementation being tested.
  const p = [.095, -.5, .5, 0, 0], w = .04;
  const psi = (x: number) => .5 * (x - .5) ** 2 - .03;
  const primitive = (x: number) => (x - .5) ** 3 / 6 - .03 * x;
  const crossings = [.5 - Math.sqrt(.1), .5 + Math.sqrt(.1), .5 - Math.sqrt(.02), .5 + Math.sqrt(.02)];
  for (const [a, b] of [[0, 1], [.137, .641], [.01, .1]] as const) {
    near(integrateDefinedDensity(p, w, a, b), clippedIntegral(psi, primitive, crossings, w, a, b), 2e-13);
  }
  const tangent = polynomialRoots([.25, -1, 1]);
  assert.equal(tangent.length, 1); near(tangent[0]!, .5, 1e-14);
});

test("clamped plane translates through former saturation with exact zero, slope and same-field mass", () => {
  const offset = .013, psi = (x: number) => x + offset, primitive = (x: number) => .5 * x * x + offset * x;
  const crossings = [-offset - width / 2, -offset + width / 2];
  const exactIntegral = (a: number, b: number) => clippedIntegral(psi, primitive, crossings, width, a, b);
  let field = initializeLatentLine(32, -1, 1, width, psi, () => 1, exactIntegral);
  let displacement = 0;
  for (let step = 0; step < 3; step++) {
    const moved = transportLatentLine(field, .037, { firstCell: 1, maximumPotentialResidual: 2e-12, maximumDerivativeResidual: 2e-11 });
    field = moved.field; displacement += .037;
    assert.ok(moved.maximumMeanResidual < 2e-13);
    near(total(field), exactIntegral(field.lower - displacement, field.lower + field.h * field.bubble.length - displacement), 2e-12);
    const zeros = roots(field); assert.equal(zeros.length, 1); near(zeros[0]!, displacement - offset);
    near(sampleLatentLine(field, zeros[0]!)[1], 1); near(sampleLatentLine(field, zeros[0]!)[2], .5);
    assertC1(field);
    assert.ok(Math.max(...field.bubble.map(Math.abs)) < 2e-13);
  }
  // The original coordinate lies deep in the original dry saturation region.
  near(sampleLatentLine(field, displacement - offset)[0], 0);
  assert.equal(definedDensity(psi(displacement - offset), width), 0);
});

test("global quadratic and quartic latent shapes survive saturated cells by transporting the current midpoint", () => {
  for (const degree of [2, 4]) {
    const scale = .2, offset = -.05, psi = (x: number) => scale * x ** degree + offset;
    const slope = (x: number) => degree * scale * x ** (degree - 1);
    const primitive = (x: number) => scale * x ** (degree + 1) / (degree + 1) + offset * x;
    const crossings = [-1, 1].flatMap(sign => [-width / 2, width / 2].map(level => sign * ((level - offset) / scale) ** (1 / degree)));
    const exactIntegral = (a: number, b: number) => clippedIntegral(psi, primitive, crossings, width, a, b);
    let field = initializeLatentLine(32, -1, 1, width, psi, slope, exactIntegral), displacement = 0;
    const initialMass = total(field);
    for (let step = 0; step < 3; step++) {
      const move = transportLatentLine(field, .019, { firstCell: 1, maximumPotentialResidual: 3e-12, maximumDerivativeResidual: 5e-11 });
      field = move.field; displacement += .019;
      near(total(field), initialMass, 3e-12); assertC1(field);
      for (let i = 0; i < 100; i++) {
        const x = field.lower + (.13 + i / 101) * field.h * field.bubble.length / 1.13;
        const actual = sampleLatentLine(field, x);
        near(actual[0], psi(x - displacement), 3e-12); near(actual[1], slope(x - displacement), 5e-11);
      }
      const zero = ((-offset) / scale) ** (1 / degree), actualRoots = roots(field);
      assert.equal(actualRoots.length, 2); near(actualRoots[0]!, displacement - zero, 3e-12); near(actualRoots[1]!, displacement + zero, 3e-12);
    }
    if (degree === 4) {
      const expectedDelta = scale * field.h ** 4 / 30;
      for (const delta of field.bubble) near(delta, expectedDelta, 3e-13);
      assert.ok(expectedDelta > 1e-8, "the pure-cell preferred latent DOF is nonzero and must not be erased");
    }
  }
});

test("pure targets reject incompatible endpoint values or inward slopes; partial mass has a finite nonlinear solution", () => {
  assert.throws(() => solveLatentCell(0, 0, .1, 0, 0, width), /pure dry target infeasible/);
  assert.throws(() => solveLatentCell(width / 2, -1, .1, 0, 0, width), /pure dry target infeasible/);
  assert.throws(() => solveLatentCell(-width / 2, 1, -.1, 0, 1, width), /pure full target infeasible/);
  assert.throws(() => solveLatentCell(.1, 0, .1, 0, -1e-5, width), /invalid nonlinear/);
  const dry = solveLatentCell(.1, 0, .1, 0, 0, width, .03);
  assert.equal(dry.delta, .03); assert.equal(dry.mean, 0);
  const sameMassOtherLatent = solveLatentCell(.1, 0, .1, 0, 0, width, .07);
  assert.equal(sameMassOtherLatent.mean, 0); assert.notEqual(sameMassOtherLatent.delta, dry.delta);
  const partial = solveLatentCell(.1, 0, .1, 0, .23, width);
  assert.ok(partial.delta < 0); near(partial.mean, .23, 2.1e-13);
});

test("saturated nonquadratic current-field orbits stay bounded and conservative with measured shape convergence", () => {
  const reports: { n: number; potentialError: number; densityError: number; slopeError: number; zeroError: number;
    maximumMassError: number; maximumResidual: number; time: number }[] = [];
  for (const n of [8, 16, 32]) {
    const begin = performance.now();
    let field = saturatedSine(n), maximumResidual = 0, maximumMassError = 0;
    const initialMass = total(field);
    for (let step = 0; step < 3 * n; step++) {
      const result = transportLatentLine(field, 1 / (3 * n), { maximumPotentialResidual: 1e-3, maximumDerivativeResidual: .1 });
      maximumResidual = Math.max(maximumResidual, result.maximumPotentialResidual); field = result.field;
      maximumMassError = Math.max(maximumMassError, Math.abs(total(field) - initialMass));
      near(total(field), initialMass, 5e-11); assertC1(field);
      assert.ok(result.maximumMeanResidual < 2.1e-13);
    }
    let potentialError = 0, densityError = 0, slopeError = 0;
    for (let i = 0; i < 2048; i++) {
      const x = (i + .317) / 2048, exactPsi = amplitude * Math.sin(tau * x), actual = sampleLatentLine(field, x);
      potentialError = Math.max(potentialError, Math.abs(actual[0] - exactPsi));
      densityError = Math.max(densityError, Math.abs(actual[2] - definedDensity(exactPsi, width)));
      slopeError = Math.max(slopeError, Math.abs(actual[1] - amplitude * tau * Math.cos(tau * x)));
      assert.ok(actual[2] >= 0 && actual[2] <= 1);
    }
    near(integrateLatentLine(field, 0, 1), initialMass, 5e-11);
    const actualRoots = roots(field), circularDistance = (a: number, b: number) => Math.min(Math.abs(a - b), Math.abs(a - b - 1), Math.abs(a - b + 1));
    const zeroError = Math.max(...actualRoots.map(root => Math.min(circularDistance(root, 0), circularDistance(root, .5))));
    for (const expected of [0, .5]) assert.ok(actualRoots.some(root => circularDistance(root, expected) < 2e-3));
    for (const root of actualRoots) near(sampleLatentLine(field, root)[2], .5, 2e-11);
    reports.push({ n, potentialError, densityError, slopeError, zeroError, maximumMassError, maximumResidual, time: performance.now() - begin });
  }
  console.log("latent nonlinear conservative orbit", JSON.stringify(reports));
  for (let i = 0; i + 1 < reports.length; i++) {
    assert.ok(reports[i]!.potentialError / reports[i + 1]!.potentialError > 8, JSON.stringify(reports));
    assert.ok(reports[i]!.densityError / reports[i + 1]!.densityError > 8, JSON.stringify(reports));
    assert.ok(reports[i]!.slopeError / reports[i + 1]!.slopeError > 4, JSON.stringify(reports));
  }
  assert.ok(reports[2]!.densityError < 1e-4 && reports[2]!.potentialError < 1e-5, JSON.stringify(reports));
});

test("transport rejects shape error or missing departure support and never changes the source", () => {
  const source = saturatedSine(8), before = [source.value.slice(), source.derivative.slice(), source.bubble.slice(), source.amount.slice()];
  assert.throws(() => transportLatentLine(source, .037, { maximumPotentialResidual: 1e-15 }), /shape residual rejected/);
  assert.deepEqual([source.value, source.derivative, source.bubble, source.amount], before);
  const open = { ...source, periodic: false };
  assert.throws(() => transportLatentLine(open, .037, { maximumPotentialResidual: 1 }), /missing current latent support/);
});
