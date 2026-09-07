/** Analytically initialized representation ladder; this does not infer shape from cell masses.
 * Reference functions use physical equations and integrals, never field.ts evaluators.
 * The surface is the rho = 1/2 level set. All fields remain in [0,1] on their box.
 */
import { polynomial, type Box, type DensityField, type Vec3 } from './field';

export interface LadderFixture {
  id: string;
  description: string;
  field: DensityField;
  box: Box;
  points: readonly Vec3[];
  exactDensity: (p: Vec3) => number;
  exactGradient: (p: Vec3) => Vec3 | null;
  exactMean: (box: Box) => number;
  surfacePoints: readonly Vec3[];
}
const box: Box = { lower: [-1, -1, -1], upper: [1, 1, 1] };
const physicalFrame = { origin: [0, 0, 0] as Vec3, scale: [1, 1, 1] as Vec3 };
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const centre = (b: Box): Vec3 => [0, 1, 2].map(a => (b.lower[a] + b.upper[a]) / 2) as unknown as Vec3;
const values = [-0.813, -0.417, -0.079, 0.263, 0.691];
const points: Vec3[] = values.flatMap(x => values.flatMap(y => values.map(z => [x, y, z] as Vec3)));
const pairs = values.flatMap(s => values.map(t => [s, t] as const));
const inside = (p: Vec3) => p.every(x => Math.abs(x) <= 1);
const affine = (offset: number, a: Vec3) => polynomial(physicalFrame, [offset, ...a, 0, 0, 0, 0, 0, 0]);

function plane(id: string, normal: Vec3, offset: number): LadderFixture {
  const slope = normal.map(x => 0.12 * x) as unknown as Vec3;
  const axis = normal.map(Math.abs).indexOf(Math.max(...normal.map(Math.abs)));
  const other = [0, 1, 2].filter(a => a !== axis);
  const surfacePoints = pairs.map(([s, t]) => {
    const p = [0, 0, 0]; p[other[0]] = s; p[other[1]] = t;
    p[axis] = (offset - normal[other[0]] * s - normal[other[1]] * t) / normal[axis];
    return p as unknown as Vec3;
  }).filter(inside);
  return { id, description: 'Arbitrary-axis affine density with an off-grid planar half-density surface.',
    box, points, field: affine(0.5 - 0.12 * offset, slope), surfacePoints,
    exactDensity: p => 0.5 + 0.12 * (dot(normal, p) - offset),
    exactGradient: () => slope,
    exactMean: b => 0.5 + 0.12 * (dot(normal, centre(b)) - offset) };
}

// Orthonormal frame tilted relative to every grid axis. Its quadratic expansion
// contains xy, xz and yz terms, while the reference stays in geometric form.
const n: Vec3 = [1 / Math.sqrt(14), 2 / Math.sqrt(14), 3 / Math.sqrt(14)];
const u: Vec3 = [2 / Math.sqrt(5), -1 / Math.sqrt(5), 0];
const v: Vec3 = [3 / Math.sqrt(70), 6 / Math.sqrt(70), -5 / Math.sqrt(70)];
function curved(id: string, k: number, l: number): LadderFixture {
  const c = [0.5, ...n.map(x => 0.12 * x),
    ...[0, 1, 2].map(a => -0.12 * (k * u[a] ** 2 + l * v[a] ** 2)),
    ...[[0, 1], [0, 2], [1, 2]].map(([a, b]) => -0.24 * (k * u[a] * u[b] + l * v[a] * v[b]))];
  const secondMoment = (direction: Vec3, b: Box) => dot(direction, centre(b)) ** 2
    + direction.reduce((sum, d, a) => sum + d * d * (b.upper[a] - b.lower[a]) ** 2 / 12, 0);
  return { id, description: 'Subtle rotated quadratic patch with all mixed terms retained.',
    box, points, field: polynomial(physicalFrame, c),
    exactDensity: p => 0.5 + 0.12 * (dot(n, p) - k * dot(u, p) ** 2 - l * dot(v, p) ** 2),
    exactGradient: p => n.map((x, a) => 0.12 * (x - 2 * k * dot(u, p) * u[a]
      - 2 * l * dot(v, p) * v[a])) as unknown as Vec3,
    exactMean: b => 0.5 + 0.12 * (dot(n, centre(b)) - k * secondMoment(u, b) - l * secondMoment(v, b)),
    surfacePoints: pairs.map(([s, t]) => u.map((x, a) => x * s + v[a] * t + n[a] * (k * s * s + l * t * t)) as unknown as Vec3).filter(inside) };
}

function edge(id: string, sign: -1 | 1, axis: number): LadderFixture {
  // Arbitrarily slanted faces, but an axis-aligned branch switch permits an
  // independent stable |x-shift| integral on every axis-aligned query box.
  const shift = 0.137, creaseSlope = 0.057;
  const a: Vec3 = axis === 0 ? [0.023, -0.039, 0.16] : [0.16, -0.039, 0.023];
  const solveAxis = axis === 0 ? 2 : 0;
  const freeAxis = 1;
  const branch = (s: number) => affine(0.5 - s * creaseSlope * shift,
    a.map((x, j) => x + (j === axis ? s * creaseSlope : 0)) as unknown as Vec3);
  const edgePairs = [...pairs, ...values.map(t => [shift, t] as const)];
  const surfacePoints = edgePairs.map(([s, t]) => {
    const p = [0, 0, 0]; p[axis] = s; p[freeAxis] = t;
    p[solveAxis] = -(a[axis] * s + a[freeAxis] * t + sign * creaseSlope * Math.abs(s - shift)) / a[solveAxis];
    return p as unknown as Vec3;
  });
  const tie: Vec3 = axis === 0 ? [shift, 0.263, -0.417] : [-0.417, 0.263, shift];
  return { id, description: `${sign < 0 ? 'Convex' : 'Concave'} sharp edge of the rho >= 1/2 region; two slanted affine faces, axis-aligned branch switch.`,
    box, points: [...points, tie], surfacePoints, field: { kind: sign < 0 ? 'minimum' : 'maximum', branches: [branch(1), branch(-1)] },
    exactDensity: p => 0.5 + dot(a, p) + sign * creaseSlope * Math.abs(p[axis] - shift),
    exactGradient: p => p[axis] === shift ? null : a.map((x, j) => x + (j === axis ? sign * creaseSlope * Math.sign(p[axis] - shift) : 0)) as unknown as Vec3,
    exactMean: b => {
      const lo = b.lower[axis] - shift, hi = b.upper[axis] - shift;
      const absMean = (hi * Math.abs(hi) - lo * Math.abs(lo)) / (2 * (hi - lo));
      return 0.5 + dot(a, centre(b)) + sign * creaseSlope * absMean;
    } };
}
export const fixtures: readonly LadderFixture[] = [
  plane('plane-oblique', [0.37, -0.61, 0.83], 0.113),
  plane('plane-nearly-axis', [0.019, 0.997, -0.041], -0.087),
  plane('plane-negative-normal', [-0.79, 0.31, -0.53], 0.041),
  curved('curved-shallow-bowl', 0.033, 0.021),
  curved('curved-shallow-saddle', 0.028, -0.019),
  curved('curved-nearly-flat', 0.0013, 0.0007),
  edge('edge-convex-x', -1, 0), edge('edge-concave-x', 1, 0),
  edge('edge-convex-z', -1, 2), edge('edge-concave-z', 1, 2),
];
export const deferredExpectations = [
  { id: 'three-face-corner', reason: 'Requires at least three active branches; two-branch records cannot represent the corner exactly.' },
  { id: 'disconnected-subcell-components', reason: 'Requires richer topology and retained component information; cell means alone cannot identify components.' },
  { id: 'rotated-edge-switch', reason: 'Representation supports affine switches, but this ladder does not yet independently validate oblique switch integration.' },
  { id: 'shape-inference-from-means', reason: 'This ladder initializes exact coefficients and does not establish feature recovery from sampled cell integrals.' },
] as const;
