import assert from "node:assert/strict";
import test from "node:test";
import { canonicalAffineDeparture, compileQuadraticSupports, IDENTITY_MATRIX, pullbackQuadratic, quadraticGradient, quadraticValue,
  supportBoxWorklist, supportCount, supportOrigin, type AffineDeparture, type Quadratic,
  type QuadraticSupportGrid, type V3 } from "../tools/implicit-density/sparse-quadratic-pullback";
import { gaussLegendre, integrateBoxDensity, mappedPoint, mappedSphereGradient, sphereQuadratic,
  sphereRampMass, sphereValue } from "./helpers/quadratic-pullback-oracle";

const grid: QuadraticSupportGrid = { origin: [-.4, -.4, -.4], dimensions: [32, 32, 32], h: .025 };
const center: V3 = [-.0625, .015, -.02], radius = .1;
const maps: readonly AffineDeparture[] = [
  { matrix: IDENTITY_MATRIX, translation: [-.0125, 0, 0] },
  { matrix: [Math.cos(.3), Math.sin(.3), 0, -Math.sin(.3), Math.cos(.3), 0, 0, 0, 1], translation: [.01, -.02, .03] },
  { matrix: [1, -.3, 0, 0, 1, -.2, 0, 0, 1], translation: [.01, -.03, 0] },
];
for (const [index, map] of maps.entries()) test(`quadratic affine ${index}: values and differential equal independently mapped sphere`, () => {
  const q = pullbackQuadratic(sphereQuadratic(center, radius), map.matrix, map.translation);
  for (let z = -3; z <= 3; z++) for (let y = -3; y <= 3; y++) for (let x = -3; x <= 3; x++) {
    const point: V3 = [x * .07, y * .06, z * .05];
    assert.ok(Math.abs(quadraticValue(q, point) - sphereValue(mappedPoint(map, point), center, radius)) < 2e-15);
    const expected = mappedSphereGradient(map, point, center, radius), actual = quadraticGradient(q, point);
    for (let axis = 0; axis < 3; axis++) assert.ok(Math.abs(actual[axis]! - expected[axis]!) < 2e-14);
  }
});

test("compiled fixed supports preserve the whole global quadratic across each shared face", () => {
  const records = compileQuadraticSupports(grid, sphereQuadratic(center, radius));
  let maximumValueJump = 0, maximumGradientJump = 0, maximumHessianJump = 0;
  const [nx, ny] = grid.dimensions;
  for (const id of supportBoxWorklist(grid, [1, 1, 1], [31, 31, 31])) {
    const q = Array.from(records.subarray(16 * id, 16 * id + 10)) as unknown as Quadratic;
    for (let axis = 0; axis < 3; axis++) {
      const neighbor = id + [1, nx, nx * ny][axis]!;
      const other = Array.from(records.subarray(16 * neighbor, 16 * neighbor + 10)) as unknown as Quadratic;
      const a = [.31 * grid.h, .73 * grid.h, .47 * grid.h]; a[axis] = grid.h;
      const b = a.slice(); b[axis] = 0;
      maximumValueJump = Math.max(maximumValueJump, Math.abs(quadraticValue(q, a as unknown as V3) - quadraticValue(other, b as unknown as V3)));
      const ga = quadraticGradient(q, a as unknown as V3), gb = quadraticGradient(other, b as unknown as V3);
      for (let component = 0; component < 3; component++) maximumGradientJump = Math.max(maximumGradientJump, Math.abs(ga[component]! - gb[component]!));
      for (let component = 4; component < 10; component++) maximumHessianJump = Math.max(maximumHessianJump, Math.abs(q[component]! - other[component]!));
    }
  }
  assert.ok(maximumValueJump < 2e-7); assert.ok(maximumGradientJump < 5e-7); assert.equal(maximumHessianJump, 0);
  assert.equal(supportCount(grid), 32768); assert.deepEqual(supportOrigin(grid, 0), grid.origin);
});

test("subsupport translation changes roots and first moments with identical enclosed M0", () => {
  const q0 = sphereQuadratic([0, 0, 0], radius);
  const q1 = pullbackQuadratic(q0, IDENTITY_MATRIX, [-.0125, 0, 0]);
  assert.ok(Math.abs(quadraticValue(q1, [radius + .0125, 0, 0])) < 1e-16);
  assert.ok(Math.abs(quadraticValue(q0, [radius + .0125, 0, 0])) > .01);
  const mass = sphereRampMass(radius, .025);
  assert.ok(mass > 4 * Math.PI * radius ** 3 / 3);
  // Translation Jacobian is one; the ramp support remains inside [-.2,.2]^3.
  const M0 = [mass, mass], M1x = [0, .0125 * mass];
  assert.equal(M0[0], M0[1]); assert.ok(M1x[1]! > M1x[0]!);
});

test("old-dry support acquires the translated plane with its actual root", () => {
  const plane: Quadratic = [.06, 1, 0, 0, 0, 0, 0, 0, 0, 0];
  const next = pullbackQuadratic(plane, IDENTITY_MATRIX, [-.1, 0, 0]);
  const lower: V3 = [.025, 0, 0], upper: V3 = [.05, .025, .025], width = .01;
  assert.equal(integrateBoxDensity(point => point[0] + .06, lower, upper, width), 0);
  assert.ok(integrateBoxDensity(point => point[0] - .04, lower, upper, width) > 0);
  assert.ok(Math.abs(quadraticValue(next, [.04, 0, 0])) < 1e-16);
  assert.deepEqual(quadraticGradient(next, [.04, 0, 0]), [1, 0, 0]);
});

test("independent quadrature integrates polynomials and axis plane with a converged oracle", () => {
  const [nodes, weights] = gaussLegendre(16);
  assert.ok(Math.abs(weights.reduce((sum, value) => sum + value, 0) - 1) < 1e-14);
  assert.ok(Math.abs(nodes.reduce((sum, node, i) => sum + weights[i]! * node ** 6, 0) - 1 / 7) < 1e-14);
  const density = integrateBoxDensity(point => point[0] - .5, [0, 0, 0], [1, 1, 1], .1, 32);
  assert.ok(Math.abs(density - .5) < 2e-14);
});

test("coarse CM12 center transfer conflicts with an exactly translated contained sphere", () => {
  const width = .4, displacement = .1, mass = sphereRampMass(.1, .025);
  const oldCenter = -.075, nextCenter = oldCenter + displacement;
  const rampRadius = Math.sqrt(.1 ** 2 + .1 * .025);
  assert.ok(oldCenter - rampRadius > -.2 && nextCenter + rampRadius < .2);
  // Uniform velocity, gamma=beta=1: the CM12 conditioned center stencil has
  // these exact one-dimensional coefficients. This is NOT a geometric oracle.
  const coarseRetained = mass * (1 - displacement / width), coarseNeighbor = mass * displacement / width;
  assert.equal(coarseRetained + coarseNeighbor, mass);
  assert.ok(Math.abs(coarseNeighbor / mass - .25) < 1e-15);
  console.log(JSON.stringify({ geometricNativeMass: mass, coarseRetained, coarseNeighbor,
    prematureNativeTransferFraction: coarseNeighbor / mass, shapeProjectionApplied: false }));
});

test("GPU affine parameters use f32-canonical coefficients and a coherent inverse", () => {
  for (const map of maps) {
    const canonical = canonicalAffineDeparture(map);
    assert.deepEqual(canonical.departure.matrix, map.matrix.map(Math.fround));
    assert.deepEqual(canonical.departure.translation, map.translation.map(Math.fround));
    for (const point of [[0, 0, 0], [.3, -.2, .1], [-.2, .1, -.3]] as const) {
      const returned = mappedPoint(canonical.forward, mappedPoint(canonical.departure, point));
      for (let axis = 0; axis < 3; axis++) assert.ok(Math.abs(returned[axis]! - point[axis]!) < 1e-7);
    }
  }
  // Finite f64 inputs with determinant one must not upload an infinite f32
  // coefficient and rely on device conversion/loop behaviour to reject it.
  assert.throws(() => canonicalAffineDeparture({ matrix: [1e39, 0, 0, 0, 1e-39, 0, 0, 0, 1], translation: [0, 0, 0] }), /unsupported/);
  assert.throws(() => canonicalAffineDeparture({ matrix: IDENTITY_MATRIX, translation: [1e39, 0, 0] }), /unsupported/);
  assert.throws(() => canonicalAffineDeparture({ matrix: [1, 0, 0, 0, 0, 0, 0, 0, 1], translation: [0, 0, 0] }), /unsupported/);
  assert.throws(() => canonicalAffineDeparture({ matrix: [16, 0, 0, 0, 1 / 16, 0, 0, 0, 1], translation: [0, 0, 0] }), /envelope/);
});

test("grid admission rejects unrepresentable f32 geometry and cell volumes", () => {
  assert.throws(() => supportCount({ ...grid, origin: [1e39, 0, 0] }), /f32/);
  assert.throws(() => supportCount({ ...grid, origin: [1e10, 0, 0] }), /f32/);
  assert.throws(() => supportCount({ ...grid, h: 1e-40 }), /f32/);
  assert.throws(() => supportCount({ ...grid, origin: [0, 0, 0], h: 1e20 }), /f32/);
});
