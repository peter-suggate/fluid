import assert from "node:assert/strict";
import test from "node:test";
import { retainedAffineRamp, retainedAffineFeature, evaluateRetainedAffineDensity, integrateRetainedAffineDensity,
  meanRetainedAffineDensity, splitRetainedAffineDensity } from "../lib/methods/adaptive-mass/sparse-cm12-retained-affine-density";

type Point = readonly [number, number, number];
const near = (actual: number, expected: number, tolerance = 2e-11) => assert.ok(Math.abs(actual - expected) <= tolerance,
  `${actual} != ${expected}, error=${Math.abs(actual - expected)}`);
// Independent 1D antiderivative of a descending unit-clamped ramp. Unlike the
// implementation's convex-polytope integral, this integrates its three pieces.
function rampPrimitive(y: number, height: number, width: number) {
  const low = height - width / 2, high = height + width / 2;
  if (y <= low) return y;
  if (y >= high) return height;
  const d = y - low; return low + d - d * d / (2 * width);
}

for (const supportWidth of [.125, 1, 8]) for (const fraction of [.25, .30, .75])
  for (const transitionFraction of [.1, 1e-5]) test(`retained ramp exact waterline/integrals: support=${supportWidth} fill=${fraction} ramp=${transitionFraction}`, () => {
    const h = supportWidth * fraction, transition = supportWidth * transitionFraction;
    const field = retainedAffineRamp({ origin: [0, 0, 0], normal: [0, 1, 0], offset: h, transitionWidth: transition });
    const box = { lower: [0, 0, 0] as Point, upper: [supportWidth, supportWidth, supportWidth] as Point };
    const before = JSON.stringify(field);
    near(evaluateRetainedAffineDensity(field, [.31 * supportWidth, h, .79 * supportWidth]), .5);
    near(meanRetainedAffineDensity(field, box), fraction);
    // Finite-volume cuts intentionally cross the ramp and do not align to it.
    for (const cut of [.13, fraction - transitionFraction / 4, fraction + transitionFraction / 4, .83]) {
      const y = supportWidth * cut;
      const lower = { lower: box.lower, upper: [supportWidth, y, supportWidth] as Point };
      const upper = { lower: [0, y, 0] as Point, upper: box.upper };
      const expected = (rampPrimitive(y, h, transition) - rampPrimitive(0, h, transition)) * supportWidth ** 2;
      near(integrateRetainedAffineDensity(field, lower), expected, 2e-11 * Math.max(1, supportWidth ** 3));
      near(integrateRetainedAffineDensity(field, lower) + integrateRetainedAffineDensity(field, upper), fraction * supportWidth ** 3,
        2e-11 * Math.max(1, supportWidth ** 3));
    }
    const partition = [box];
    for (let cycle = 0; cycle < 12; cycle++) {
      const pick = (cycle * 7) % partition.length;
      const children = splitRetainedAffineDensity(field, partition[pick]!);
      assert.ok(children.every(child => child.field === field), "splits retain the exact immutable authority; no refit");
      partition.splice(pick, 1, ...children.map(child => child.box));
      near(partition.reduce((sum, cell) => sum + integrateRetainedAffineDensity(field, cell), 0), fraction * supportWidth ** 3,
        3e-11 * Math.max(1, supportWidth ** 3));
    }
    assert.equal(JSON.stringify(field), before);
  });

for (const direction of [1, -1]) for (const transition of [.08, 1e-6])
  test(`oblique signed normal=${direction} narrow ramp=${transition}: exact plane and independent slab integrals`, () => {
    // Height h=.35+.2x+.1z stays clear of top/bottom. The exact volume of
    // every full-height x/z slab is its area times the affine mean height.
    const norm = Math.hypot(.2, 1, .1);
    const field = retainedAffineRamp({ origin: [0, 0, 0], normal: [-.2 * direction, direction, -.1 * direction],
      offset: direction * .35 / norm, transitionWidth: transition / norm });
    for (let i = 0; i < 19; i++) {
      const x = (i + .3) / 20, z = ((i * 7) % 19 + .7) / 20, h = .35 + .2 * x + .1 * z;
      near(evaluateRetainedAffineDensity(field, [x, h, z]), .5, 3e-10);
      const x0 = i / 20, x1 = (i + 1) / 20;
      const box = { lower: [x0, 0, .13] as Point, upper: [x1, 1, .87] as Point };
      const heightMean = .35 + .2 * (x0 + x1) / 2 + .1 * .5;
      near(meanRetainedAffineDensity(field, box), direction === 1 ? heightMean : 1 - heightMean, 3e-9);
      const children = splitRetainedAffineDensity(field, box);
      near(children.reduce((sum, child) => sum + integrateRetainedAffineDensity(child.field, child.box), 0),
        (direction === 1 ? heightMean : 1 - heightMean) * (x1 - x0) * .74, 3e-10);
    }
  });

for (const kind of ["minimum", "maximum"] as const) test(`${kind}: exact retained crease without a fitted heightfield`, () => {
  const n = Math.hypot(.2, 1), width = .02;
  const left = retainedAffineRamp({ origin: [0, 0, 0], normal: [-.2, 1, 0], offset: .25 / n, transitionWidth: width / n });
  const right = retainedAffineRamp({ origin: [0, 0, 0], normal: [.2, 1, 0], offset: .45 / n, transitionWidth: width / n });
  const feature = retainedAffineFeature(kind, [left, right]);
  const sign = kind === "minimum" ? -1 : 1;
  const box = { lower: [0, 0, 0] as Point, upper: [1, 1, 1] as Point };
  for (let i = 0; i <= 20; i++) {
    const x = i / 20, y = .35 + sign * .2 * Math.abs(x - .5);
    near(evaluateRetainedAffineDensity(feature, [x, y, .37]), .5);
  }
  near(meanRetainedAffineDensity(feature, box), .35 + sign * .05);
  near(splitRetainedAffineDensity(feature, box).reduce((amount, child) =>
    amount + integrateRetainedAffineDensity(child.field, child.box), 0), .35 + sign * .05);
});
