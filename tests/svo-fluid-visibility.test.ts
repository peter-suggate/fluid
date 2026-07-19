import assert from "node:assert/strict";
import test from "node:test";

import {
  estimateSvoFluidGradient,
  findNearestSvoFluidSignChange,
  refineSvoFluidZero,
  resolveSvoFluidPhi,
  svoFluidSamplesCrossZero,
  svoFluidVisibilityWGSL,
  traceSvoFluidLevelSet,
  type SvoFluidFieldPair,
  type SvoFluidOwnedSample,
} from "../lib/svo-fluid-visibility";

const coarse = (phi_m: number, valid = true): SvoFluidFieldPair => ({ coarse: { phi_m, valid } });
const owned = (phi_m: number, t_m: number, owner: "coarse" | "fine" = "coarse") => ({
  phi_m, t_m, owner, valid: true,
} as const);

test("fine phi exclusively owns valid samples and invalid fine data falls back to coarse", () => {
  assert.deepEqual(resolveSvoFluidPhi({
    coarse: { phi_m: -0.4, valid: true },
    fine: { phi_m: 0.1, valid: true },
  }), { phi_m: 0.1, valid: true, owner: "fine" });
  assert.deepEqual(resolveSvoFluidPhi({
    coarse: { phi_m: -0.4, valid: true },
    fine: { phi_m: Number.NaN, valid: true },
  }), { phi_m: -0.4, valid: true, owner: "coarse" });
  assert.deepEqual(resolveSvoFluidPhi({
    coarse: { phi_m: 0, valid: false },
    fine: { phi_m: 0, valid: false },
  }), { phi_m: Number.POSITIVE_INFINITY, valid: false, owner: "none" });
});

test("nearest sign-change detection is boundary-inclusive and never bridges invalid samples", () => {
  const invalid: SvoFluidOwnedSample = { phi_m: Number.NaN, valid: false, owner: "none" };
  assert.equal(svoFluidSamplesCrossZero(owned(-1, 0), owned(1, 1)), true);
  assert.equal(svoFluidSamplesCrossZero(owned(1, 0), owned(0, 1)), true);
  assert.equal(svoFluidSamplesCrossZero(owned(1, 0), invalid), false);
  assert.deepEqual(findNearestSvoFluidSignChange([
    owned(1, 0), { ...invalid, t_m: 1 }, owned(-1, 2), owned(1, 3, "fine"), owned(-1, 4),
  ]), { lower: owned(-1, 2), upper: owned(1, 3, "fine") });
  assert.deepEqual(findNearestSvoFluidSignChange([owned(1, 0), owned(0, 0.5), owned(-1, 1)]), {
    lower: owned(0, 0.5), upper: owned(0, 0.5),
  });
});

test("safeguarded secant and bisection refinement remain bounded", () => {
  const interval = { lower: owned(-0.75, 0), upper: owned(0.25, 2) };
  const linear = refineSvoFluidZero(interval, (t_m) => ({ phi_m: t_m - 1.5, valid: true, owner: "fine" }), {
    maximumIterations: 8,
    tTolerance_m: 1e-9,
    phiTolerance_m: 1e-9,
  });
  assert.equal(linear.status, "hit");
  if (linear.status === "hit") {
    assert.ok(Math.abs(linear.sample.t_m - 1.5) < 1e-9);
    assert.equal(linear.sample.owner, "fine");
    assert.ok(linear.iterations <= 8);
  }

  const nonlinear = refineSvoFluidZero(
    { lower: owned(-1, 0), upper: owned(7, 2) },
    (t_m) => ({ phi_m: t_m ** 3 - 1, valid: true, owner: "coarse" }),
    { maximumIterations: 4, tTolerance_m: 1e-12, phiTolerance_m: 1e-12 },
  );
  assert.equal(nonlinear.status, "hit");
  if (nonlinear.status === "hit") {
    assert.equal(nonlinear.iterations, 4);
    assert.equal(nonlinear.converged, false);
    assert.ok(nonlinear.sample.t_m >= 0 && nonlinear.sample.t_m <= 2);
  }
});

test("inside-fluid rays find the outgoing interface instead of treating the origin as a miss", () => {
  const result = traceSvoFluidLevelSet(
    { origin_m: [0, 0, 0], direction: [4, 0, 0], tMax_m: 4 },
    ([x]) => coarse(x - 2),
    { cellSize_m: [0.25, 0.5, 1], step_m: 0.4, maximumSteps: 16 },
  );
  assert.equal(result.status, "hit");
  if (result.status === "hit") {
    assert.equal(result.insideFluidAtStart, true);
    assert.ok(Math.abs(result.t_m - 2) < 1e-5);
    assert.deepEqual(result.position_m, [2, 0, 0]);
    assert.deepEqual(result.normal, [1, 0, 0]);
  }
});

test("anisotropic cell widths produce a world-space gradient and valid fine ownership", () => {
  const plane = ([x, y, z]: readonly [number, number, number]): SvoFluidFieldPair => ({
    coarse: { phi_m: x + 2 * y - 0.5 * z - 3, valid: true },
    fine: { phi_m: x + 2 * y - 0.5 * z - 3, valid: x >= 0 },
  });
  const gradient = estimateSvoFluidGradient([2, 1, 2], [0.125, 0.75, 2.5], plane);
  assert.equal(gradient.valid, true);
  assert.equal(gradient.scheme, "central");
  assert.ok(Math.abs(gradient.gradient[0] - 1) < 1e-12);
  assert.ok(Math.abs(gradient.gradient[1] - 2) < 1e-12);
  assert.ok(Math.abs(gradient.gradient[2] + 0.5) < 1e-12);

  const hit = traceSvoFluidLevelSet(
    { origin_m: [0, 1, 2], direction: [1, 0, 0], tMax_m: 4 },
    plane,
    { cellSize_m: [0.125, 0.75, 2.5], step_m: 0.3 },
  );
  assert.equal(hit.status, "hit");
  if (hit.status === "hit") assert.equal(hit.fieldOwner, "fine");
});

test("non-finite and degenerate gradients use a finite deterministic fallback", () => {
  const degenerate = estimateSvoFluidGradient([0, 0, 0], [0.2, 0.5, 1], () => coarse(-0.2), {
    fallbackNormal: [0, 0, -4],
  });
  assert.deepEqual(degenerate, {
    normal: [0, 0, -1], gradient: [0, 0, 0], valid: false, scheme: "fallback",
  });

  const nonFinite = estimateSvoFluidGradient([0, 0, 0], [0.2, 0.5, 1], ([x]) => ({
    coarse: { phi_m: x < 0 ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY, valid: true },
  }), { fallbackNormal: [0, 0, 0] });
  assert.equal(nonFinite.valid, false);
  assert.deepEqual(nonFinite.normal, [0, 1, 0]);
  assert.ok(nonFinite.gradient.every(Number.isFinite));
});

test("trace reports bounded exhaustion and invalid fields distinctly", () => {
  assert.deepEqual(traceSvoFluidLevelSet(
    { origin_m: [0, 0, 0], direction: [1, 0, 0], tMax_m: 10 },
    () => coarse(1),
    { cellSize_m: [1, 1, 1], step_m: 1, maximumSteps: 2 },
  ), { status: "work-exhausted", steps: 2, insideFluidAtStart: false });
  assert.deepEqual(traceSvoFluidLevelSet(
    { origin_m: [0, 0, 0], direction: [1, 0, 0], tMax_m: 1 },
    () => ({ coarse: { phi_m: 0, valid: false } }),
    { cellSize_m: [1, 1, 1] },
  ), { status: "invalid-field", steps: 0, insideFluidAtStart: false });
});

test("WGSL helpers are binding-free, bounded, anisotropic, and keep fluid ownership explicit", () => {
  assert.match(svoFluidVisibilityWGSL, /fn svoResolveFluidPhi/);
  assert.match(svoFluidVisibilityWGSL, /SVO_FLUID_OWNER_FINE/);
  assert.match(svoFluidVisibilityWGSL, /fine\.valid!=0u[\s\S]*SVO_FLUID_OWNER_FINE/);
  assert.match(svoFluidVisibilityWGSL, /fn svoFluidCrossesZero/);
  assert.match(svoFluidVisibilityWGSL, /fn svoFluidSecantOrBisect/);
  assert.match(svoFluidVisibilityWGSL, /SVO_FLUID_REFINE_ITERATIONS:u32 = 8u/);
  assert.match(svoFluidVisibilityWGSL, /2\.0\*cellSize_m\.x/);
  assert.match(svoFluidVisibilityWGSL, /2\.0\*cellSize_m\.y/);
  assert.match(svoFluidVisibilityWGSL, /2\.0\*cellSize_m\.z/);
  assert.match(svoFluidVisibilityWGSL, /fallback=-rayDirection/);
  assert.doesNotMatch(svoFluidVisibilityWGSL, /solidPhi|combinedPhi|solidField/);
  assert.doesNotMatch(svoFluidVisibilityWGSL, /@group|@binding/);
});
