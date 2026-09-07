/** Standalone research algebra. No production solver imports this module.
 * Densities are means of a retained physical-coordinate field, not point data.
 */
export type Vec3 = readonly [number, number, number];
export interface Box { readonly lower: Vec3; readonly upper: Vec3 }
export interface Frame { readonly origin: Vec3; readonly scale: Vec3 }
/** Basis: 1, x, y, z, xx, yy, zz, xy, xz, yz, in the record's local frame. */
export type Coefficients = readonly [number, number, number, number, number,
  number, number, number, number, number];
export interface Polynomial { readonly kind: "polynomial"; readonly frame: Frame;
  readonly coefficients: Coefficients }
export interface Edge { readonly kind: "minimum" | "maximum";
  readonly branches: readonly [Polynomial, Polynomial] }
export type DensityField = Polynomial | Edge;
export const COEFFICIENT_COUNT = 10;
const crossAxes = [[0, 1], [0, 2], [1, 2]] as const;

export function validateBox(box: Box): void {
  for (let axis = 0; axis < 3; axis++) if (!Number.isFinite(box.lower[axis])
    || !Number.isFinite(box.upper[axis]) || box.upper[axis] <= box.lower[axis]) {
    throw new Error("expected a finite positive-volume box");
  }
}
export function volume(box: Box): number {
  return (box.upper[0] - box.lower[0]) * (box.upper[1] - box.lower[1])
    * (box.upper[2] - box.lower[2]);
}
export function frameForBox(box: Box): Frame {
  validateBox(box);
  return { origin: box.lower.map((lo, a) => (lo + box.upper[a]) / 2) as unknown as Vec3,
    scale: box.lower.map((lo, a) => box.upper[a] - lo) as unknown as Vec3 };
}
export function polynomial(frame: Frame, coefficients: readonly number[]): Polynomial {
  if (coefficients.length !== 10 || !coefficients.every(Number.isFinite)
    || !frame.origin.every(Number.isFinite)
    || !frame.scale.every(x => Number.isFinite(x) && x > 0)) {
    throw new Error("invalid density polynomial");
  }
  return { kind: "polynomial", frame: { origin: [...frame.origin], scale: [...frame.scale] },
    coefficients: [...coefficients] as unknown as Coefficients };
}
export function localCoordinates(frame: Frame, point: Vec3): Vec3 {
  return point.map((x, a) => (x - frame.origin[a]) / frame.scale[a]) as unknown as Vec3;
}
export function basisAt([x, y, z]: Vec3): Coefficients {
  return [1, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z];
}
export function meanBasis(frame: Frame, box: Box): Coefficients {
  const centre = box.lower.map((lo, a) => (lo + box.upper[a]) / 2) as unknown as Vec3;
  const [x, y, z] = localCoordinates(frame, centre);
  const w = box.lower.map((lo, a) => (box.upper[a] - lo) / frame.scale[a]);
  return [1, x, y, z, x * x + w[0] ** 2 / 12, y * y + w[1] ** 2 / 12,
    z * z + w[2] ** 2 / 12, x * y, x * z, y * z];
}
export function evaluatePolynomial(field: Polynomial, point: Vec3): number {
  const b = basisAt(localCoordinates(field.frame, point));
  return field.coefficients.reduce((sum, c, i) => sum + c * b[i], 0);
}
export function evaluate(field: DensityField, point: Vec3): number {
  if (field.kind === "polynomial") return evaluatePolynomial(field, point);
  const a = evaluatePolynomial(field.branches[0], point);
  const b = evaluatePolynomial(field.branches[1], point);
  return field.kind === "minimum" ? Math.min(a, b) : Math.max(a, b);
}
export function gradient(field: Polynomial, point: Vec3): Vec3 {
  const x = localCoordinates(field.frame, point), c = field.coefficients;
  const g = [c[1] + 2 * c[4] * x[0], c[2] + 2 * c[5] * x[1], c[3] + 2 * c[6] * x[2]];
  crossAxes.forEach(([a, b], k) => { g[a] += c[7 + k] * x[b]; g[b] += c[7 + k] * x[a]; });
  return g.map((v, a) => v / field.frame.scale[a]) as unknown as Vec3;
}
export function meanPolynomial(field: Polynomial, box: Box): number {
  const b = meanBasis(field.frame, box);
  return field.coefficients.reduce((sum, c, i) => sum + c * b[i], 0);
}

/** Exact coordinate change. It neither resamples means nor fits new normals. */
export function rebase(field: Polynomial, frame: Frame): Polynomial {
  const s = frame.scale.map((w, i) => w / field.frame.scale[i]);
  const c = field.coefficients;
  const g = gradient(field, frame.origin);
  return polynomial(frame, [evaluatePolynomial(field, frame.origin),
    g[0] * frame.scale[0], g[1] * frame.scale[1], g[2] * frame.scale[2],
    c[4] * s[0] ** 2, c[5] * s[1] ** 2, c[6] * s[2] ** 2,
    c[7] * s[0] * s[1], c[8] * s[0] * s[2], c[9] * s[1] * s[2]]);
}
export function rebaseField(field: DensityField, frame: Frame): DensityField {
  return field.kind === "polynomial" ? rebase(field, frame)
    : { kind: field.kind, branches: [rebase(field.branches[0], frame), rebase(field.branches[1], frame)] };
}
export function splitBox(box: Box): Box[] {
  const middle = frameForBox(box).origin;
  return Array.from({ length: 8 }, (_, octant) => ({
    lower: box.lower.map((lo, a) => octant & (1 << a) ? middle[a] : lo) as unknown as Vec3,
    upper: box.upper.map((hi, a) => octant & (1 << a) ? hi : middle[a]) as unknown as Vec3,
  }));
}

/** Integrate max(affine,0) using a truncated-simplex antiderivative.
 * Numerical cancellation for extremely oblique/slender boxes remains a GPU
 * validation requirement. Coefficients near zero are not discarded here.
 */
export function positiveAffineMean(field: Polynomial, box: Box): number {
  if (field.coefficients.slice(4).some(c => c !== 0)) throw new Error("affine integration requires an affine branch");
  const c = rebase(field, { origin: box.lower, scale: frameForBox(box).scale }).coefficients;
  let lower = c[0]; const widths: number[] = [];
  for (let axis = 0; axis < 3; axis++) {
    lower += Math.min(c[1 + axis], 0);
    if (c[1 + axis] !== 0) widths.push(Math.abs(c[1 + axis]));
  }
  const upper = lower + widths.reduce((a, b) => a + b, 0);
  if (upper <= 0) return 0;
  if (lower >= 0) return meanPolynomial(field, box);
  const n = widths.length;
  let total = 0;
  for (let mask = 0; mask < 1 << n; mask++) {
    let x = upper, sign = 1;
    widths.forEach((w, a) => { if (mask & (1 << a)) { x -= w; sign = -sign; } });
    total += sign * Math.max(0, x) ** (n + 1);
  }
  const factorial = [1, 1, 2, 6, 24][n + 1];
  return total / (factorial * widths.reduce((a, b) => a * b, 1));
}
export function mean(field: DensityField, box: Box): number {
  if (field.kind === "polynomial") return meanPolynomial(field, box);
  const a = field.branches[0], b = rebase(field.branches[1], a.frame);
  if (a.coefficients.slice(4).some(x => x !== 0) || b.coefficients.slice(4).some(x => x !== 0)) {
    throw new Error("this first oracle integrates two affine feature branches only");
  }
  const delta = polynomial(a.frame, a.coefficients.map((c, k) => c - b.coefficients[k]));
  const positive = positiveAffineMean(delta, box);
  return field.kind === "minimum" ? meanPolynomial(a, box) - positive
    : meanPolynomial(b, box) + positive;
}
/** Physical transition width belongs to field coefficients, never to query-cell size. */
export function clampedAffineMean(field: Polynomial, box: Box): number {
  const shifted = polynomial(field.frame, field.coefficients.map((c, k) => k === 0 ? c - 1 : c));
  return positiveAffineMean(field, box) - positiveAffineMean(shifted, box);
}
export function splitField(field: DensityField, box: Box) {
  return splitBox(box).map(child => ({ box: child, density: mean(field, child),
    field: rebaseField(field, frameForBox(child)) }));
}

/** Conservative admission rule for this algebraic family: merge only fields
 * whose rebased coefficients agree. Averaged means alone never authorize it.
 * Caller supplies the covering partition; this helper does not prove coverage.
 */
export function mergeEquivalentFields(fields: readonly DensityField[], frame: Frame,
  tolerance = 1e-12): DensityField | null {
  if (!fields.length || !(tolerance >= 0) || !Number.isFinite(tolerance)) return null;
  const rebased = fields.map(f => rebaseField(f, frame));
  const first = rebased[0];
  const coefficients = (f: DensityField): readonly number[] => f.kind === "polynomial"
    ? f.coefficients : [...f.branches[0].coefficients, ...f.branches[1].coefficients];
  const expected = coefficients(first);
  return rebased.every(f => f.kind === first.kind && coefficients(f).every((c, i) =>
    Math.abs(c - expected[i]) <= tolerance)) ? first : null;
}
