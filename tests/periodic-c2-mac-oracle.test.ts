import assert from "node:assert/strict";
import test from "node:test";
import { cardinalSpline, FACE_FILTER, PeriodicC2MacOracle, discreteMacDivergence, macIndex,
  type Triple, type MacWords, type PeriodicMacGrid } from "../tools/implicit-density/periodic-c2-mac-oracle";

const grid: PeriodicMacGrid = { dimensions: [4, 5, 6], h: .17, origin: [-.32, .13, -.21] };
const count = grid.dimensions.reduce((a, b) => a * b, 1);
const data = (phase = 0): MacWords => [0, 1, 2].map(axis => Float64Array.from({ length: count }, (_, id) =>
  Math.sin(id * 1.789 + axis * 2.371 + phase) + .23 * Math.cos(id * 2.31 + axis))) as unknown as MacWords;
const near = (a: number, b: number, tol = 2e-10) => assert.ok(Number.isFinite(a) && Math.abs(a - b) < tol, `${a} != ${b} (${tol})`);
const points: Triple[] = Array.from({ length: 19 }, (_, i) => grid.origin.map((o, axis) =>
  o + grid.h * (grid.dimensions[axis]! * ((i * .173 + axis * .213 + .127) % 1))) as unknown as Triple);

test("centered spline chain is normalized; the common face filter has a bounded invertible symbol", () => {
  for (const degree of [3, 4]) for (const x of [-.73, 0, .19, .5, 1.28]) {
    let sum = 0; for (let k = -5; k <= 5; k++) sum += cardinalSpline(degree, x - k);
    near(sum, 1, 2e-15);
  }
  for (let offset = -2; offset <= 2; offset++) near(cardinalSpline(4, offset), FACE_FILTER[offset + 2]!, 2e-15);
  const symbol = (theta: number) => FACE_FILTER[2] + 2 * FACE_FILTER[3] * Math.cos(theta) + 2 * FACE_FILTER[4] * Math.cos(2 * theta);
  near(symbol(0), 1, 2e-15); near(symbol(Math.PI), 5 / 24, 2e-15);
  // Positive quadratic in cos(theta), increasing on [-1,1]: exact min at pi.
  assert.ok(2 * FACE_FILTER[3] - 8 * FACE_FILTER[4] > 0);
});

test("continuous divergence equals B3 interpolation of discrete divergence for arbitrary coefficients", () => {
  const coefficients = data(), field = PeriodicC2MacOracle.fromSplineCoefficients(grid, coefficients);
  const div = discreteMacDivergence(grid, coefficients), n = grid.dimensions;
  for (const p of points) {
    const q = p.map((x, axis) => (x - grid.origin[axis]!) / grid.h - .5);
    let expected = 0;
    // Independent complete periodic lattice image sum, not the query stencil.
    for (let k = -3; k < n[2] + 3; k++) for (let j = -3; j < n[1] + 3; j++) for (let i = -3; i < n[0] + 3; i++) {
      expected += cardinalSpline(3, q[0]! - i) * cardinalSpline(3, q[1]! - j) * cardinalSpline(3, q[2]! - k)
        * div[macIndex(n, i, j, k)]!;
    }
    near(field.divergence(p), expected);
  }
});

function curlFaces(): MacWords {
  const a = data(.31), n = grid.dimensions;
  const result = [new Float64Array(count), new Float64Array(count), new Float64Array(count)] as const;
  for (let k = 0; k < n[2]; k++) for (let j = 0; j < n[1]; j++) for (let i = 0; i < n[0]; i++) {
    const id = macIndex(n, i, j, k), dx = (c: number) => (a[c]![id]! - a[c]![macIndex(n, i - 1, j, k)]!) / grid.h;
    const dy = (c: number) => (a[c]![id]! - a[c]![macIndex(n, i, j - 1, k)]!) / grid.h;
    const dz = (c: number) => (a[c]![id]! - a[c]![macIndex(n, i, j, k - 1)]!) / grid.h;
    result[0][id] = dy(2) - dz(1); result[1][id] = dz(0) - dx(2); result[2][id] = dx(1) - dy(0);
  }
  return result;
}
test("common periodic prefilter retains solenoidal data and reproduces physical face averages", () => {
  const faces = curlFaces(), field = PeriodicC2MacOracle.fromFaceAverages(grid, faces), n = grid.dimensions;
  assert.ok(Math.max(...discreteMacDivergence(grid, faces).map(Math.abs)) < 1e-12);
  for (const p of points) near(field.divergence(p), 0, 3e-10);
  // Independent exact quadrature: split at the transverse cubic knots, then
  // Gauss3 integrates each polynomial subrectangle. This does not use K.
  const nodes = [-Math.sqrt(3 / 5), 0, Math.sqrt(3 / 5)], weights = [5 / 9, 8 / 9, 5 / 9];
  for (let component = 0; component < 3; component++) for (const ijk of [[0, 0, 0], [3, 4, 5], [1, 2, 3]]) {
    let average = 0; const u = (component + 1) % 3, v = (component + 2) % 3;
    for (let tile = 0; tile < 4; tile++) for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
      const p = ijk.map((c, axis) => grid.origin[axis]! + grid.h * (c + (axis === component ? 1 : 0)));
      p[u]! += grid.h * (.5 * (tile & 1) + .25 * (1 + nodes[i]!));
      p[v]! += grid.h * (.5 * (tile >> 1) + .25 * (1 + nodes[j]!));
      average += weights[i]! * weights[j]! * field.component(p as unknown as Triple, component) / 16;
    }
    near(average, faces[component]![macIndex(n, ijk[0]!, ijk[1]!, ijk[2]!)]!, 2e-10);
  }
});

test("C2 field jets agree at a representative knot on each axis and its periodic image; constants stay constant", () => {
  const field = PeriodicC2MacOracle.fromFaceAverages(grid, data());
  const derivatives: Triple[] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]];
  const epsilon = 1e-8;
  for (let axis = 0; axis < 3; axis++) for (let component = 0; component < 3; component++) {
    const p = [...points[3]!]; p[axis] = grid.origin[axis]! + 1.5 * grid.h;
    const left = [...p], right = [...p]; left[axis]! -= epsilon; right[axis]! += epsilon;
    for (const derivative of derivatives) {
      const center = field.component(p as unknown as Triple, component, derivative);
      const error = Math.max(Math.abs(field.component(left as unknown as Triple, component, derivative) - center),
        Math.abs(field.component(right as unknown as Triple, component, derivative) - center));
      // C2 limits: halve the distance and observe first-order approach, even
      // for Hessians whose third derivative may jump at this knot.
      left[axis] = p[axis]! - epsilon / 2; right[axis] = p[axis]! + epsilon / 2;
      const halfError = Math.max(Math.abs(field.component(left as unknown as Triple, component, derivative) - center),
        Math.abs(field.component(right as unknown as Triple, component, derivative) - center));
      assert.ok(halfError <= .501 * error + 1e-10);
      left[axis] = p[axis]! - epsilon; right[axis] = p[axis]! + epsilon;
      const wrapped = [...p]; wrapped[axis]! += grid.h * grid.dimensions[axis]!;
      near(field.component(wrapped as unknown as Triple, component, derivative), center, 2e-10);
    }
  }
  const constants = [1.23, -.71, .07];
  const constant = PeriodicC2MacOracle.fromFaceAverages(grid,
    constants.map(v => new Float64Array(count).fill(v)) as unknown as MacWords);
  for (const p of points) for (let component = 0; component < 3; component++) for (const derivative of derivatives) {
    near(constant.component(p, component, derivative), derivative.every(v => v === 0) ? constants[component]! : 0, 2e-11);
  }
});

test("unfiltered spline coefficients smooth the face data, and divergent inputs are never projected away", () => {
  const n: Triple = [4, 4, 4], evenGrid = { ...grid, dimensions: n };
  const nyquist = Float64Array.from({ length: 64 }, (_, id) => (id % 2 ? -1 : 1)
    * (Math.floor(id / 4) % 2 ? -1 : 1) * (Math.floor(id / 16) % 2 ? -1 : 1));
  const faces: MacWords = [nyquist, nyquist.slice(), nyquist.slice()];
  const field = PeriodicC2MacOracle.fromFaceAverages(evenGrid, faces);
  near(field.coefficients[0][0]!, (24 / 5) ** 3, 3e-12);
  assert.ok(Math.max(...discreteMacDivergence(evenGrid, field.coefficients).map(Math.abs)) > 1);
  assert.throws(() => PeriodicC2MacOracle.fromFaceAverages(grid, [new Float64Array(1), faces[1], faces[2]]));
});
