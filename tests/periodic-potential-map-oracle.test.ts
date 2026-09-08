import assert from "node:assert/strict";
import test from "node:test";
import { PeriodicC2MacOracle, discreteMacDivergence, macIndex, type MacWords, type PeriodicMacGrid, type Triple } from "../tools/implicit-density/periodic-c2-mac-oracle";
import { PeriodicPotentialMapOracle, determinant3, type Matrix3 } from "../tools/implicit-density/periodic-potential-map-oracle";

const grid: PeriodicMacGrid = { dimensions: [4, 5, 4], h: .2, origin: [-.2, .1, -.3] };
const count = grid.dimensions.reduce((a, b) => a * b, 1);
const potentials = [0, 1, 2].map(axis => Float64Array.from({ length: count }, (_, id) =>
  .012 * Math.sin(id * 1.713 + axis * 2.391) + .006 * Math.cos(id * 2.171 + axis))) as unknown as MacWords;
const mean: Triple = [.12, -.08, .04];
const field = new PeriodicPotentialMapOracle(grid, potentials, mean);
const points: Triple[] = [[-.07, .237, -.149], [.193, .517, .083], [.421, .839, -.017]];
const near = (a: number, b: number, tolerance = 2e-11) => assert.ok(Number.isFinite(a) && Math.abs(a - b) < tolerance,
  `${a} != ${b}; error ${Math.abs(a - b)} exceeds ${tolerance}`);
const distance = (a: readonly number[], b: readonly number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i]!)));

/** Independent index-direction differences; no production curl helper. */
function independentCurl(): MacWords {
  const n = grid.dimensions, result = [new Float64Array(count), new Float64Array(count), new Float64Array(count)] as const;
  for (let k = 0; k < n[2]; k++) for (let j = 0; j < n[1]; j++) for (let i = 0; i < n[0]; i++) {
    const id = macIndex(n, i, j, k), xm = macIndex(n, i - 1, j, k), ym = macIndex(n, i, j - 1, k), zm = macIndex(n, i, j, k - 1);
    result[0][id] = (potentials[2][id]! - potentials[2][ym]! - potentials[1][id]! + potentials[1][zm]!) / grid.h + mean[0];
    result[1][id] = (potentials[0][id]! - potentials[0][zm]! - potentials[2][id]! + potentials[2][xm]!) / grid.h + mean[1];
    result[2][id] = (potentials[1][id]! - potentials[1][xm]! - potentials[0][id]! + potentials[0][ym]!) / grid.h + mean[2];
  }
  return result;
}
const mac = PeriodicC2MacOracle.fromSplineCoefficients(grid, independentCurl());

test("periodic edge-potential curl equals the C2 MAC spline of discrete edge curl through second derivatives", () => {
  const derivativeOrders: Triple[] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]];
  const coefficients = field.curlSplineCoefficients(), reference = independentCurl();
  for (let component = 0; component < 3; component++) assert.deepEqual(coefficients[component], reference[component]);
  assert.ok(Math.max(...discreteMacDivergence(grid, coefficients).map(Math.abs)) < 1e-14);
  for (const p of points) for (const derivative of derivativeOrders) {
    const actual = field.velocity(p, derivative);
    for (let component = 0; component < 3; component++) near(actual[component]!, mac.component(p, component, derivative), 8e-12);
  }
});

function finiteJacobian(map: (p: Triple) => Triple, p: Triple, epsilon = 1e-5): Matrix3 {
  const result = new Array<number>(9);
  for (let axis = 0; axis < 3; axis++) {
    const a = [...p] as [number, number, number], b = [...p] as [number, number, number]; a[axis]! -= epsilon; b[axis]! += epsilon;
    const lo = map(a), hi = map(b);
    for (let row = 0; row < 3; row++) result[3 * row + axis] = (hi[row]! - lo[row]!) / (2 * epsilon);
  }
  return result as unknown as Matrix3;
}

test("converged pair maps preserve the frozen coordinate, reverse, and match independently differentiated Jacobians and Hessians", () => {
  let largestResidual = 0, determinantError = 0, finiteDifferenceError = 0;
  for (const p of points) for (let component = 0; component < 3; component++) {
    const result = field.pair(p, component, .11), inverse = field.pair(result.point, component, -.11);
    assert.equal(result.point[component], p[component]);
    near(distance(inverse.point, p), 0, 8e-13);
    largestResidual = Math.max(largestResidual, result.residual);
    assert.ok(result.maximumPairPositionErrorBound >= result.residual && result.maximumPairPositionErrorBound < 3e-13);
    determinantError = Math.max(determinantError, Math.abs(determinant3(result.jacobian) - 1));
    const finite = finiteJacobian(point => field.pair(point, component, .11).point, p);
    finiteDifferenceError = Math.max(finiteDifferenceError, distance(finite, result.jacobian));
    near(determinant3(finite), 1, 3e-8);
    const epsilon = 1e-5;
    for (let axis = 0; axis < 3; axis++) {
      const a = [...p] as [number, number, number], b = [...p] as [number, number, number]; a[axis]! -= epsilon; b[axis]! += epsilon;
      const lo = field.pair(a, component, .11).jacobian, hi = field.pair(b, component, .11).jacobian;
      for (let row = 0; row < 3; row++) for (let col = 0; col < 3; col++) {
        near((hi[3 * row + col]! - lo[3 * row + col]!) / (2 * epsilon), result.hessians[row]![3 * col + axis]!, 5e-7);
      }
    }
  }
  console.log("Hamiltonian pair CPU metrics", JSON.stringify({ largestResidual, determinantError, finiteDifferenceError }));
  assert.ok(largestResidual <= 2e-13 && determinantError < 2e-14 && finiteDifferenceError < 3e-8);
});

test("pair-map C2 jets approach common limits when the midpoint crosses each spline face", () => {
  const dt = .13, component = 2;
  for (let axis = 0; axis < 3; axis++) {
    const midpoint = [...points[1]!] as [number, number, number]; midpoint[axis] = grid.origin[axis]! + 1.5 * grid.h;
    const pairVelocity: Triple = [field.potential(midpoint, component, [0, 1, 0]), -field.potential(midpoint, component, [1, 0, 0]), 0];
    const start = midpoint.map((v, i) => v - .5 * dt * pairVelocity[i]!) as unknown as Triple;
    const center = field.pair(start, component, dt);
    for (let i = 0; i < 3; i++) near((start[i]! + center.point[i]!) / 2, midpoint[i]!, 2e-13);
    if (axis !== component) assert.ok((start[axis]! - midpoint[axis]!) * (center.point[axis]! - midpoint[axis]!) < 0,
      "the trajectory must cross the active-coordinate spline face");
    const flatten = (jet: ReturnType<typeof field.pair>) => [...jet.point, ...jet.jacobian, ...jet.hessians.flat()];
    const exact = flatten(center), epsilon = 1e-5;
    const errorAt = (distance: number) => [-1, 1].map(sign => {
      const p = [...start] as [number, number, number]; p[axis]! += sign * distance;
      return flatten(field.pair(p, component, dt)).map((v, i) => Math.abs(v - exact[i]!));
    });
    const whole = errorAt(epsilon), half = errorAt(epsilon / 2);
    for (let i = 0; i < exact.length; i++) assert.ok(Math.max(half[0]![i]!, half[1]![i]!) <= .505 * Math.max(whole[0]![i]!, whole[1]![i]!) + 2e-10,
      `axis=${axis}, jet=${i}`);
    const periodic = [...start] as [number, number, number]; periodic[axis]! += grid.dimensions[axis]! * grid.h;
    const translated = field.pair(periodic, component, dt);
    near(translated.point[axis]! - center.point[axis]!, grid.dimensions[axis]! * grid.h);
    near(distance(translated.jacobian, center.jacobian), 0); near(distance(translated.hessians.flat(), center.hessians.flat()), 0);
  }
});

function referenceFlow(initial: Triple, duration: number, steps: number): Triple {
  const h = duration / steps;
  let p = [...initial] as [number, number, number];
  // Independent RK4 of the equivalent MAC spline, not pair-map iteration or
  // its implicit derivative. Reference refinement is checked separately.
  const velocity = (point: Triple) => [0, 1, 2].map(axis => mac.component(point, axis)) as unknown as Triple;
  const shifted = (x: Triple, u: Triple, scale: number) => x.map((v, i) => v + scale * u[i]!) as unknown as Triple;
  for (let step = 0; step < steps; step++) {
    const a = velocity(p), b = velocity(shifted(p, a, h / 2)), c = velocity(shifted(p, b, h / 2)), d = velocity(shifted(p, c, h));
    p = p.map((v, i) => v + h * (a[i]! + 2 * b[i]! + 2 * c[i]! + d[i]!) / 6) as [number, number, number];
  }
  return p;
}

test("symmetric pair composition converges to the independently refined full velocity flow and reverses", () => {
  const begin = performance.now(), errors = [0, 0, 0], duration = .3;
  let referenceDifference = 0, inverseError = 0, maxDeterminantError = 0;
  for (const p of points) {
    const reference = referenceFlow(p, duration, 256);
    referenceDifference = Math.max(referenceDifference, distance(reference, referenceFlow(p, duration, 128)));
    for (const [rung, steps] of [2, 4, 8].entries()) {
      let moved = p;
      for (let step = 0; step < steps; step++) {
        const next = field.step(moved, duration / steps); moved = next.point;
        maxDeterminantError = Math.max(maxDeterminantError, Math.abs(determinant3(next.jacobian) - 1));
      }
      errors[rung] = Math.max(errors[rung]!, distance(moved, reference));
      for (let step = 0; step < steps; step++) moved = field.step(moved, -duration / steps).point;
      inverseError = Math.max(inverseError, distance(moved, p));
    }
    const composed = field.step(p, .11), finite = finiteJacobian(point => field.step(point, .11).point, p);
    near(distance(composed.jacobian, finite), 0, 4e-8);
    if (p === points[0]) for (let axis = 0; axis < 3; axis++) {
      const epsilon = 1e-5, a = [...p] as [number, number, number], b = [...p] as [number, number, number];
      a[axis]! -= epsilon; b[axis]! += epsilon;
      const lo = field.step(a, .11).jacobian, hi = field.step(b, .11).jacobian;
      for (let row = 0; row < 3; row++) for (let col = 0; col < 3; col++) {
        near((hi[3 * row + col]! - lo[3 * row + col]!) / (2 * epsilon), composed.hessians[row]![3 * col + axis]!, 5e-7);
      }
    }
  }
  console.log("symmetric map CPU convergence", JSON.stringify({ errors, referenceDifference, inverseError, maxDeterminantError, milliseconds: performance.now() - begin }));
  assert.ok(referenceDifference < 2e-10 && inverseError < 3e-12 && maxDeterminantError < 4e-14);
  assert.ok(errors[0]! / errors[1]! > 3.5 && errors[1]! / errors[2]! > 3.5, JSON.stringify(errors));
  assert.ok(errors[2]! < 5e-6);
});

test("unconverged, ill-conditioned or uncertified pair maps reject without changing the source", () => {
  const before = field.potentials.map(v => v.slice()), p = points[0]!;
  assert.throws(() => field.pair(p, 2, .11, { maximumIterations: 0 }), /unconverged/);
  assert.throws(() => field.pair(p, 2, .11, { maximumConditionNumber: 1 }), /conditioning rejected/);
  assert.throws(() => field.pair(p, 2, 10), /global contraction bound rejected/);
  assert.throws(() => field.pair(p, 2, Number.NaN), /invalid pair-map/);
  const stationary = field.pair(p, 2, 0, { maximumIterations: 0 });
  assert.deepEqual(stationary.point, p); assert.equal(stationary.residual, 0); assert.equal(stationary.iterations, 0);
  assert.deepEqual(field.potentials, before);
  const externalSnapshot = field.potentials; externalSnapshot[0][0] = 1e30;
  assert.deepEqual(field.potentials, before, "external QA writes cannot invalidate the cached contraction bound");
});
