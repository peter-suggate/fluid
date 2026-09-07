import assert from "node:assert/strict";
import test from "node:test";
import { evaluateRetainedAffineDensity, integrateRetainedAffineDensity,
  meanRetainedAffineDensity, retainedAffineFeature, retainedAffineRamp,
  splitRetainedAffineDensity } from "../lib/methods/adaptive-mass/sparse-cm12-retained-affine-density";
const near = (actual: number, expected: number, tolerance = 2e-12) => assert.ok(Math.abs(actual - expected) < tolerance,
  `${actual} versus ${expected}: error ${Math.abs(actual - expected)}`);
const box = { lower: [0, 0, 0] as const, upper: [1, 1, 1] as const };

for (const kind of ["minimum", "maximum"] as const) test(`retained ${kind} ramp crease: exact independent height integral and two normals`, () => {
  const norm = Math.sqrt(1.04), transitionWidth = 0.05;
  const a = retainedAffineRamp({ origin: [0, 0, 0], normal: [-0.2, 1, 0], offset: 0.4 / norm, transitionWidth });
  const b = retainedAffineRamp({ origin: [0, 0, 0], normal: [0.2, 1, 0], offset: 0.6 / norm, transitionWidth });
  const field = retainedAffineFeature(kind, [a, b]);
  near(integrateRetainedAffineDensity(field, box), kind === "minimum" ? 0.45 : 0.55);
  near(evaluateRetainedAffineDensity(field, [0.5, 0.5, 0.37]), 0.5);
  const children = splitRetainedAffineDensity(field, box);
  near(children.reduce((sum, c) => sum + c.mean / 8, 0), kind === "minimum" ? 0.45 : 0.55);
  assert.ok(children.every(c => c.field === field));
  const epsilon = 1e-5;
  const left = (evaluateRetainedAffineDensity(field, [0.5, 0.5, 0.3])
    - evaluateRetainedAffineDensity(field, [0.5 - epsilon, 0.5, 0.3])) / epsilon;
  const right = (evaluateRetainedAffineDensity(field, [0.5 + epsilon, 0.5, 0.3])
    - evaluateRetainedAffineDensity(field, [0.5, 0.5, 0.3])) / epsilon;
  near(left, (kind === "minimum" ? 1 : -1) * 0.2 / (norm * transitionWidth), 1e-9);
  near(right, -left, 1e-9);
});

for (const normal of [[1e-14, 1, -1e-12], [-1e-14, -1, 1e-12]] as const) {
  test(`near-axis ramp ${normal[1]}: independent height integral without tiny-coefficient division`, () => {
    const ramp = retainedAffineRamp({ origin: [0, 0, 0], normal, offset: normal[1] * 0.3, transitionWidth: 1e-7 });
    const height = (ramp.offset - ramp.normal[0] / 2 - ramp.normal[2] / 2) / ramp.normal[1];
    near(meanRetainedAffineDensity(ramp, box), normal[1] > 0 ? height : 1 - height);
  });
}

test("slender physical support with twelve orders of anisotropy retains the declared volume", () => {
  const lower = [2, -4, 1e4] as const, upper = [2 + 1e-9, -1, 1e4 + 1e3] as const;
  const ramp = retainedAffineRamp({ origin: lower, normal: [1e-12, 1, 1e-12], offset: 0.9, transitionWidth: 1e-6 });
  const widths = upper.map((v, i) => v - lower[i]);
  const expectedMean = (ramp.offset - ramp.normal[0] * widths[0] / 2 - ramp.normal[2] * widths[2] / 2)
    / (ramp.normal[1] * widths[1]);
  near(meanRetainedAffineDensity(ramp, { lower, upper }), expectedMean);
  near(integrateRetainedAffineDensity(ramp, { lower, upper }), expectedMean * widths[0] * widths[1] * widths[2], 1e-18);
});

test("equal branches integrate once, positive and negative branch dominance agree with direct ramps", () => {
  const low = retainedAffineRamp({ origin: [0, 0, 0], normal: [0, 1, 0], offset: 0.25, transitionWidth: 0.05 });
  const high = retainedAffineRamp({ origin: [0, 0, 0], normal: [0, 1, 0], offset: 0.75, transitionWidth: 0.05 });
  near(meanRetainedAffineDensity(retainedAffineFeature("minimum", [low, low]), box), 0.25);
  near(meanRetainedAffineDensity(retainedAffineFeature("minimum", [low, high]), box), 0.25);
  near(meanRetainedAffineDensity(retainedAffineFeature("maximum", [low, high]), box), 0.75);
});

test("fully oblique ramp agrees with independent tetrahedron affine moment", () => {
  const norm = Math.sqrt(3);
  const ramp = retainedAffineRamp({ origin: [0, 0, 0], normal: [1, 1, 1], offset: 0.5 / norm, transitionWidth: 1 / norm });
  // q=max(1-x-y-z,0): tetra volume 1/6 times average vertex value 1/4.
  near(integrateRetainedAffineDensity(ramp, box), 1 / 24);
  near(splitRetainedAffineDensity(ramp, box).reduce((sum, child) => sum + child.mean / 8, 0), 1 / 24);
});

test("cross-axis crease moments match independently integrated min(x,y) and max(x,y)", () => {
  const x = retainedAffineRamp({ origin: [0, 0, 0], normal: [-1, 0, 0], offset: -0.5, transitionWidth: 1 });
  const y = retainedAffineRamp({ origin: [0, 0, 0], normal: [0, -1, 0], offset: -0.5, transitionWidth: 1 });
  near(integrateRetainedAffineDensity(retainedAffineFeature("minimum", [x, y]), box), 1 / 3);
  near(integrateRetainedAffineDensity(retainedAffineFeature("maximum", [x, y]), box), 2 / 3);
});

test("oblique unequal-width feature branch partitions preserve min+max integral identity", () => {
  for (let i = 0; i < 31; i++) {
    const a = retainedAffineRamp({ origin: [0.3, 0.5, 0.7], normal: [Math.sin(i + 1), Math.cos(i * 3), 0.2],
      offset: 0.15, transitionWidth: 10 ** (-(i % 7)) });
    const b = retainedAffineRamp({ origin: [0.2, 0.4, 0.6], normal: [-0.7, Math.sin(i * 5), Math.cos(i * 2)],
      offset: -0.05, transitionWidth: 0.11 });
    const expected = integrateRetainedAffineDensity(a, box) + integrateRetainedAffineDensity(b, box);
    const actual = integrateRetainedAffineDensity(retainedAffineFeature("minimum", [a, b]), box)
      + integrateRetainedAffineDensity(retainedAffineFeature("maximum", [a, b]), box);
    near(actual, expected, 2e-10);
  }
});
