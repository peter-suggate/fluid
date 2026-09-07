import assert from "node:assert/strict";
import test from "node:test";
import { evaluateRetainedQuadraticDensity, gradientRetainedQuadraticDensity, integrateRetainedQuadraticDensity,
  retainedQuadraticDensity, retainedSphereDensity, type QuadraticDensityPoint } from "../lib/methods/adaptive-mass/sparse-cm12-retained-quadratic-density";
const near = (actual: number, expected: number, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) < tolerance,
  `${actual} != ${expected}`);

test("declared sphere retains analytic radius and radial gradient without a sampled surface band", () => {
  const radius = 0.5, width = 0.1, field = retainedSphereDensity([0, 0, 0], radius, width);
  assert.equal(field.coefficients.length, 10);
  for (let i = 0; i < 27; i++) {
    const theta = i * 0.729, phi = i * 0.383;
    const point: QuadraticDensityPoint = [radius * Math.cos(theta) * Math.cos(phi), radius * Math.sin(theta) * Math.cos(phi), radius * Math.sin(phi)];
    near(evaluateRetainedQuadraticDensity(field, point), 0.5);
    const g = gradientRetainedQuadraticDensity(field, point);
    g.forEach((v, axis) => near(v, -point[axis] / (radius * width)));
  }
  assert.equal(evaluateRetainedQuadraticDensity(field, [0, 0, 0]), 1);
  assert.equal(evaluateRetainedQuadraticDensity(field, [1, 0, 0]), 0);
  assert.ok(Object.isFrozen(field.coefficients));
});

test("sphere certified enclosures contain an independent radial diffuse mass and converge with budget", () => {
  const r = 0.5, w = 0.1, field = retainedSphereDensity([0, 0, 0], r, w);
  const inner = Math.sqrt(r * r - r * w), outer = Math.sqrt(r * r + r * w);
  const a = 0.5 + r / (2 * w), b = 1 / (2 * r * w);
  // Radial integral of 4*pi*r²*q(r), including saturated inner ball.
  const exact = 4 * Math.PI * (inner ** 3 / 3 + a * (outer ** 3 - inner ** 3) / 3 - b * (outer ** 5 - inner ** 5) / 5);
  const sharpVolume = 4 * Math.PI * r ** 3 / 3;
  assert.ok(Math.abs(exact - sharpVolume) > 1e-3, "diffuse mass must not silently become sharp occupancy volume");
  const box = { lower: [-1, -1, -1] as const, upper: [1, 1, 1] as const };
  let previousWidth = Infinity;
  for (const maximumLeaves of [64, 512, 4096]) {
    const receipt = integrateRetainedQuadraticDensity(field, box, { maximumLeaves, absoluteTolerance: 1e-4 });
    assert.ok(receipt.lower <= exact && exact <= receipt.upper, JSON.stringify({ exact, receipt }));
    assert.ok(receipt.upper - receipt.lower < previousWidth / 2);
    assert.equal(receipt.toleranceMet, false, "budget exhaustion must remain explicit");
    assert.equal(receipt.leaves, maximumLeaves);
    previousWidth = receipt.upper - receipt.lower;
  }
});

test("unclamped shallow quadratic integrates in one interval box and preserves its analytic surface", () => {
  const field = retainedQuadraticDensity({ origin: [0, 0, 0], scale: [1, 1, 1],
    coefficients: [0.5, 0, -0.3, 0, 0.02, 0, -0.01, 0, 0, 0] });
  const receipt = integrateRetainedQuadraticDensity(field, { lower: [-0.5, -0.5, -0.5], upper: [0.5, 0.5, 0.5] },
    { absoluteTolerance: 1e-12, maximumLeaves: 1 });
  assert.equal(receipt.toleranceMet, true); assert.equal(receipt.leaves, 1);
  const exact = 0.5 + 0.01 / 12;
  assert.ok(receipt.lower <= exact && exact <= receipt.upper);
  near(receipt.estimate, exact);
  for (const x of [-0.4, 0.1, 0.43]) for (const z of [-0.3, 0.2]) {
    const y = (0.02 * x * x - 0.01 * z * z) / 0.3;
    near(evaluateRetainedQuadraticDensity(field, [x, y, z]), 0.5);
  }
});

test("clamped shallow bowl converges to independent height integral without retaining a finest grid", () => {
  // q=clamp(.5+(.3+.05*(x-.5)^2-y)/.05,0,1).
  const field = retainedQuadraticDensity({ origin: [0, 0, 0], scale: [1, 1, 1],
    coefficients: [6.75, -1, -20, 0, 1, 0, 0, 0, 0, 0] });
  const exact = 0.3 + 0.05 / 12, box = { lower: [0, 0, 0] as const, upper: [1, 1, 1] as const };
  const receipt = integrateRetainedQuadraticDensity(field, box, { absoluteTolerance: 1e-3, maximumLeaves: 4096 });
  assert.equal(receipt.toleranceMet, true);
  assert.ok(receipt.lower <= exact && exact <= receipt.upper);
  assert.ok(receipt.leaves < 200);
  assert.equal(field.coefficients.length, 10);
});

test("declared initializer produces retained numeric primitives without occupancy-mean reconstruction", async () => {
  const { initializeRetainedDensityPrimitive } = await import("../lib/methods/adaptive-mass/sparse-cm12-retained-quadratic-density");
  const pool = initializeRetainedDensityPrimitive({ kind: "pool", height: 0.25, transitionWidth: 0.05, generation: 3 });
  assert.equal(pool.kind, "clamped-affine"); assert.equal(pool.generation, 3);
  const sphere = initializeRetainedDensityPrimitive({ kind: "sphere", center: [1, 2, 3], radius: 2, transitionWidth: 0.1 });
  assert.equal(sphere.kind, "clamped-quadratic");
  if (sphere.kind === "clamped-quadratic") near(evaluateRetainedQuadraticDensity(sphere, [3, 2, 3]), 0.5);
  assert.ok(Object.values(sphere).every(v => typeof v !== "function"));
});
