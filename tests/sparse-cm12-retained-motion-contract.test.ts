import assert from "node:assert/strict";
import test from "node:test";
import {
  evaluateRetainedSceneDensity, integrateRetainedSceneDensity, retainedSceneDensity,
  type RetainedScenePoint,
} from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";

// Independent counterexamples to the invalid design assumption that matching M0
// determines transported geometry. These tests validate the rejection oracle;
// they do NOT claim that the production evolution kernel passes a motion gate.
// Keep these negative controls when replacing the fixed q_seed + (a,b) authority.
const close = (actual: number, expected: number, tolerance = 1e-12) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected} (tolerance ${tolerance})`);
const clamp = (x: number) => Math.max(0, Math.min(1, x));
const nativeBox = { lower: [0, 0, 0], upper: [.4, .4, .4] } as const;
const neighborBox = { lower: [.4, 0, 0], upper: [.8, .4, .4] } as const;
const sphere = (cx: number) => retainedSceneDensity({ generation: 1, transitionWidth: .05,
  domain: { lower: [-1, -1, -1], upper: [2, 2, 2] },
  primitives: [{ kind: "ellipsoid", center: [cx, .2, .2], radii: [.1, .1, .1] }] });

test("motion oracle rejects a pinned sphere even when every occupied native M0 is exact", () => {
  const previous = sphere(.15), translated = sphere(.25);
  const before = integrateRetainedSceneDensity(previous, nativeBox, { absoluteTolerance: 1e-12 });
  const after = integrateRetainedSceneDensity(translated, nativeBox, { absoluteTolerance: 1e-12 });
  assert.ok(before.toleranceMet && after.toleranceMet);
  close(before.amount, after.amount, 2e-15);
  assert.equal(integrateRetainedSceneDensity(previous, neighborBox).amount, 0);
  assert.equal(integrateRetainedSceneDensity(translated, neighborBox).amount, 0);

  // Even granting IDEAL translated native means, an M0-only update observes no
  // change. The current fixed-seed affine lift consequently keeps a=1,b=0.
  const pinnedNegativeControl = (p: RetainedScenePoint) => evaluateRetainedSceneDensity(previous, p);
  const desired = (p: RetainedScenePoint) => evaluateRetainedSceneDensity(translated, p);
  const samples: RetainedScenePoint[] = [[.075, .2, .2], [.325, .2, .2]];
  const motionError = Math.max(...samples.map(p => Math.abs(pinnedNegativeControl(p) - desired(p))));
  assert.ok(motionError > .9, "the geometry oracle must reject the equal-mass pinned field");
  const previousPrimitive = previous.primitives[0], nextPrimitive = translated.primitives[0];
  assert.ok(previousPrimitive.kind === "ellipsoid" && nextPrimitive.kind === "ellipsoid");
  close(nextPrimitive.center[0] - previousPrimitive.center[0], .1, 1e-8);
});

test("native center interpolation conserves M0 but contradicts a contained subcell translation", () => {
  const previous = sphere(.15), translated = sphere(.25);
  const mass = integrateRetainedSceneDensity(previous, nativeBox).amount;
  const exactNext = [integrateRetainedSceneDensity(translated, nativeBox).amount,
    integrateRetainedSceneDensity(translated, neighborBox).amount];

  // Interior uniform velocity, uniform H, gamma=beta=1: CM12's backward linear
  // stencil reduces to the standard conservative fractional translation of
  // means. Column and row sums are already one, so conditioning cannot alter it.
  const fraction = .1 / .4;
  const centerInterpolated = [(1 - fraction) * mass, fraction * mass];
  close(centerInterpolated[0] + centerInterpolated[1], mass);
  close(exactNext[0], mass); assert.equal(exactNext[1], 0);
  close(centerInterpolated[1] / mass, .25);
  assert.ok(centerInterpolated[1] > .001, "a shape update cannot obey this target and exact translation together");
});

test("an initially dry fine support can gain the exact plane mass and still lack its interface", () => {
  const h = .05, oldHeight = -h, nextHeight = .375 * h;
  const exact = (y: number) => clamp(.5 - (y - nextHeight) / h);
  const old = (y: number) => clamp(.5 - (y - oldHeight) / h);
  const exactMean = .5 * .875 * .875; // Triangle of height .875 and base .875h.
  close(exactMean, .3828125);
  for (const y of [0, nextHeight, h]) assert.equal(old(y), 0);
  close(exact(nextHeight), .5);
  assert.ok(exact(0) > .5 && exact(h) < .5);

  // Any a*q_seed+b on this support is constant. Choosing b=the exact new mean
  // passes its mass restriction but cannot create the required interior root.
  const massExactNegativeControl = (_y: number) => exactMean;
  close(massExactNegativeControl(0) * h, .5 * .875 * (.875 * h));
  for (let i = 0; i <= 100; i++) assert.ok(massExactNegativeControl(i * h / 100) < .5);
  assert.ok(Math.abs(massExactNegativeControl(nextHeight) - .5) > .1);
});

test("an affine rescaling of the old ramp cannot turn the normal of a falling sphere", () => {
  const radius = .25, fall = .1, width = .05;
  const oldRelativePoint = [radius, -fall, 0];
  const oldPhi = (radius * radius + fall * fall - radius * radius) / (2 * radius);
  const oldDensity = clamp(.5 - oldPhi / width);
  assert.ok(oldDensity > 0 && oldDensity < 1, "the derivative argument uses an unsaturated old ramp");
  const length = Math.hypot(...oldRelativePoint);
  const oldNormal = oldRelativePoint.map(x => x / length);
  const desiredNormal = [1, 0, 0]; // New equator, relative to the translated center.
  const angle = Math.acos(oldNormal.reduce((sum, x, axis) => sum + x * desiredNormal[axis], 0));
  close(angle * 180 / Math.PI, 21.80140948635181, 1e-12);
  // For positive a, grad(a*q_seed+b) is parallel to grad(q_seed). a=0 instead
  // destroys the normal. No choice of the two scalar coefficients solves this.
  assert.ok(angle > .38);
  close(Math.sqrt(radius * radius + radius * width) - radius, .023861278752583037);
});

test("transported spatial moments detect translation that native M0 cannot observe", () => {
  const field = sphere(.15), primitive = field.primitives[0]; assert.ok(primitive.kind === "ellipsoid");
  const mass = integrateRetainedSceneDensity(field, nativeBox).amount;
  const center = [.15, .2, .2], shift = [.1, 0, 0];
  // Independent radial integration of r^2*q gives the diffuse sphere's actual
  // isotropic covariance, not the sharp-sphere R^2/5 or its momentum channels.
  const r = primitive.radii[0], w = field.transitionWidth;
  const inner = Math.sqrt(r * r - r * w), outer = Math.sqrt(r * r + r * w);
  const radialSecond = 4 * Math.PI * (inner ** 5 / 5
    + ((r * r + r * w) * (outer ** 5 - inner ** 5) / 5 - (outer ** 7 - inner ** 7) / 7) / (2 * r * w));
  const variance = radialSecond / (3 * mass), first = center.map(x => mass * x);
  assert.ok(variance > r * r / 5, "the diffuse shell contributes to the spatial moment");
  const second = center.map((x, i) => center.map((y, j) => mass * (x * y + (i === j ? variance : 0))));
  const nextCenter = center.map((x, i) => x + shift[i]);
  const nextFirst = nextCenter.map(x => mass * x);
  const nextSecond = nextCenter.map((x, i) => nextCenter.map((y, j) => mass * (x * y + (i === j ? variance : 0))));
  for (let i = 0; i < 3; i++) {
    close(nextFirst[i], first[i] + shift[i] * mass);
    for (let j = 0; j < 3; j++) close(nextSecond[i][j], second[i][j]
      + shift[i] * first[j] + first[i] * shift[j] + shift[i] * shift[j] * mass);
  }
  close((nextFirst[0] - first[0]) / mass, .1);
});

test("M0, centroid and covariance do not uniquely determine a general half-density surface", () => {
  // Legendre P3 is orthogonal to 1,x,x^2. These admissible [0,1] fields have
  // identical first three moments but different half-density sets. A moment
  // scheme must therefore bound representation error; ten 3D moments alone
  // cannot certify an arbitrary curved or branched interface.
  const p3 = (x: number) => (5 * x ** 3 - 3 * x) / 2;
  const plus = (x: number) => .6 + .2 * p3(x), minus = (x: number) => .6 - .2 * p3(x);
  const nodes = [-Math.sqrt(3 / 5), 0, Math.sqrt(3 / 5)], weights = [5 / 9, 8 / 9, 5 / 9];
  for (let power = 0; power <= 2; power++) {
    const moment = (q: (x: number) => number) => nodes.reduce((sum, x, i) => sum + weights[i] * x ** power * q(x), 0);
    close(moment(plus), moment(minus));
    close(moment(plus), [1.2, 0, .4][power]);
  }
  for (let i = 0; i <= 100; i++) {
    const x = -1 + i / 50;
    assert.ok(plus(x) >= .4 - 1e-15 && plus(x) <= .8 + 1e-15);
    assert.ok(minus(x) >= .4 - 1e-15 && minus(x) <= .8 + 1e-15);
  }
  assert.ok(plus(.95) > .5 && minus(.95) < .5);
  assert.ok(plus(-.95) < .5 && minus(-.95) > .5);
});
