import assert from "node:assert/strict";
import test from "node:test";
import { densityFromPhi, jacobianDeterminant, midpointSaddleDeparture, NONLINEAR_SHEAR,
  pulledBall, pulledBoxBranches, shearDeparture, shearDepartureJacobian, shearForward,
  type Jacobian3, type Point3 } from "../tools/implicit-density/volume-preserving-map-oracle";

function derivative(f: (p: Point3) => Point3, p: Point3, h = 1e-5): Jacobian3 {
  const result = new Array<number>(9);
  for (let axis = 0; axis < 3; axis++) {
    const a = [...p] as [number, number, number], b = [...a] as [number, number, number];
    a[axis] -= h; b[axis] += h;
    const lo = f(a), hi = f(b);
    for (let row = 0; row < 3; row++) result[3 * row + axis] = (hi[row] - lo[row]) / (2 * h);
  }
  return result as unknown as Jacobian3;
}
function distance(a: readonly number[], b: readonly number[]) {
  return Math.max(...a.map((value, index) => Math.abs(value - b[index])));
}

test("nonlinear three-shear fixture is invertible and volume preserving off lattice", () => {
  let maxDerivativeError = 0, maxNumericalDetError = 0;
  for (let i = 0; i < 96; i++) {
    const p: Point3 = [Math.sin(1.3 * i), .3 + Math.cos(.7 * i), .7 * Math.sin(2.1 * i)];
    assert.ok(distance(shearDeparture(shearForward(p)), p) < 8e-16);
    assert.ok(distance(shearForward(shearDeparture(p)), p) < 8e-16);
    const exact = shearDepartureJacobian(p), numerical = derivative(shearDeparture, p);
    assert.ok(Math.abs(jacobianDeterminant(exact) - 1) < 5e-16);
    maxDerivativeError = Math.max(maxDerivativeError, distance(exact, numerical));
    maxNumericalDetError = Math.max(maxNumericalDetError, Math.abs(jacobianDeterminant(numerical) - 1));
  }
  assert.ok(maxDerivativeError < 2e-9, `${maxDerivativeError}`);
  assert.ok(maxNumericalDetError < 2e-9, `${maxNumericalDetError}`);
  // It must actually deform: the map is not another affine test in disguise.
  const a: Point3 = [.15, .31, .47], b: Point3 = [.25, .31, .47], c: Point3 = [.35, .31, .47];
  assert.ok(Math.abs(shearForward(a)[2] - 2 * shearForward(b)[2] + shearForward(c)[2]) > .01);
});

test("curved density and intentional box branches have analytic transported normals", () => {
  const center: Point3 = [.5, .5, .5], radius = .15, width = .05;
  for (let i = 0; i < 24; i++) {
    const angle = i * Math.PI / 12;
    const material: Point3 = [.5 + radius * Math.cos(angle), .5 + radius * Math.sin(angle), .5];
    const p = shearForward(material), sample = pulledBall(p, center, radius);
    assert.ok(Math.abs(sample.phi) < 2e-16);
    assert.ok(Math.abs(densityFromPhi(sample.phi, width) - .5) < 4e-15);
    const numeric = derivative(point => [pulledBall(point, center, radius).phi, 0, 0], p);
    assert.ok(distance(sample.gradient, numeric.slice(0, 3)) < 5e-9);
  }
  const lower: Point3 = [.35, .35, .35], upper: Point3 = [.65, .65, .65];
  const edge = shearForward([.65, .65, .5]), branches = pulledBoxBranches(edge, lower, upper);
  const active = branches.filter(branch => Math.abs(branch.phi) < 1e-14);
  assert.deepEqual(active.map(branch => branch.id), [1, 3]);
  assert.ok(distance(active[0].gradient, active[1].gradient) > .5);
  for (const branch of active) {
    const numeric = derivative(point => [pulledBoxBranches(point, lower, upper)[branch.id].phi, 0, 0], edge);
    assert.ok(distance(branch.gradient, numeric.slice(0, 3)) < 2e-9);
  }
  // Reversal recovers sharp branch identity, not an averaged or fitted normal.
  assert.ok(distance(shearDeparture(edge), [.65, .65, .5]) < 2e-16);
  assert.equal(NONLINEAR_SHEAR.amplitudes.length, 3);
});

test("divergence-free velocity alone does not conserve mass under midpoint pullback", () => {
  const rate = 2, dt = .2, steps = 20;
  const columns = [midpointSaddleDeparture([1, 0, 0], rate, dt),
    midpointSaddleDeparture([0, 1, 0], rate, dt), midpointSaddleDeparture([0, 0, 1], rate, dt)];
  const j = columns[0].map((_, row) => columns.map(column => column[row])).flat() as unknown as Jacobian3;
  const measuredDet = jacobianDeterminant(j), expectedDet = 1 + (rate * dt) ** 4 / 4;
  assert.ok(Math.abs(measuredDet - expectedDet) < 5e-16);
  assert.ok(measuredDet > 1.006);
  // Integrating q(Bx) over all space changes any integrable source's M0 by
  // 1/det(B). No discrete density resampling is needed for this failure.
  const massRatio = measuredDet ** -steps;
  assert.ok(massRatio < .89 && massRatio > .87, `mass ratio ${massRatio}`);
  // Exact continuous saddle flow has reciprocal scales and unit determinant.
  const exact: Jacobian3 = [Math.exp(-rate * dt), 0, 0, 0, Math.exp(rate * dt), 0, 0, 0, 1];
  assert.ok(Math.abs(jacobianDeterminant(exact) - 1) < 2e-16);
});
