import assert from "node:assert/strict";
import test from "node:test";
import { affineSupportQ, sphereBoxAmount, spherePhi, sphereQ, sphereTotalAmount,
  type Point, type Sphere } from "../tools/retained-imposed-flow-oracle";
const s: Sphere = { center: [-.15, .8, 0], radius: .25, width: .05 };
test("independent sphere integration partitions, translates and agrees with radial shell integral", () => {
  let amount = 0, error = 0;
  // One exact-y slab plus 8x8 rectangles avoids asking an adaptive rule to
  // discover a compact shape inside an arbitrarily large empty domain.
  for (let x = 0; x < 8; x++) for (let z = 0; z < 8; z++) {
    const lower: Point = [-.55 + .1 * x, .4, -.4 + .1 * z];
    const upper: Point = [lower[0] + .1, 1.2, lower[2] + .1];
    const r = sphereBoxAmount(s, lower, upper, 1e-10);
    const shift = .025;
    const moved = sphereBoxAmount({ ...s, center: [s.center[0] + shift, .8, 0] },
      [lower[0] + shift, lower[1], lower[2]], [upper[0] + shift, upper[1], upper[2]], 1e-10);
    assert.ok(Math.abs(r.amount - moved.amount) < 1e-14);
    amount += r.amount; error += r.estimatedError;
  }
  assert.ok(error < 1e-8);
  assert.ok(Math.abs(amount - sphereTotalAmount(s)) < 1e-9);
});
test("sphere point and gradient references follow a half-cell translation", () => {
  const translated = { ...s, center: [-.125, .8, 0] as Point };
  const p: Point = [.125, .8, 0];
  assert.equal(spherePhi(translated, p), 0); assert.equal(sphereQ(translated, p), .5);
  assert.equal(sphereQ(s, p), 0);
  // Every affine rescaling remains constant on an originally dry support.
  const constant = affineSupportQ(s, p, .2, .3);
  assert.equal(constant, .3); assert.ok(Math.abs(constant - sphereQ(translated, p)) > .19);
});
