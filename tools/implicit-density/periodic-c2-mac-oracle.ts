/** CPU mathematical oracle, deliberately not a production velocity compiler.
 * Uniform periodic MAC face averages -> C2 divergence-compatible velocity.
 * Positive faces of cell (i,j,k) store their normal velocity averaged over
 * the complete face. These are NOT collocated VEX samples or liquid fluxes.
 *
 * B4 in the component's normal direction and B3 transversely form a spline
 * derivative chain. Their face averages apply the SAME periodic B4 filter in
 * every direction/component. A common inverse filter therefore preserves
 * discrete divergence while reproducing the input face averages. The tiny
 * dense solve here is an independent reference, not a CPU runtime proposal.
 */
export type Triple = readonly [number, number, number];
export type MacWords = readonly [Float64Array, Float64Array, Float64Array];
export interface PeriodicMacGrid { dimensions: Triple; h: number; origin: Triple }
export const FACE_FILTER = [1 / 384, 76 / 384, 230 / 384, 76 / 384, 1 / 384] as const;
const wrap = (i: number, n: number) => ((i % n) + n) % n;
export const macIndex = (n: Triple, i: number, j: number, k: number) =>
  wrap(i, n[0]) + n[0] * (wrap(j, n[1]) + n[1] * wrap(k, n[2]));

/** Centered cardinal spline via compact, stable Cox-de Boor recurrence. */
export function cardinalSpline(degree: number, x: number): number {
  if (degree === 0) return x >= -.5 && x < .5 ? 1 : 0;
  if (Math.abs(x) >= (degree + 1) / 2) return 0;
  return ((x + (degree + 1) / 2) * cardinalSpline(degree - 1, x + .5)
    + ((degree + 1) / 2 - x) * cardinalSpline(degree - 1, x - .5)) / degree;
}
export function splineDerivative(degree: number, x: number, order: number): number {
  if (order === 0) return cardinalSpline(degree, x);
  return splineDerivative(degree - 1, x + .5, order - 1)
    - splineDerivative(degree - 1, x - .5, order - 1);
}

function validate(grid: PeriodicMacGrid, faces: MacWords): number {
  if (!grid.dimensions.every(n => Number.isInteger(n) && n >= 3 && n <= 16)
    || !Number.isFinite(grid.h) || grid.h <= 0 || !grid.origin.every(Number.isFinite)) {
    throw new Error("Periodic oracle requires finite geometry and 3..16 cells per axis");
  }
  const count = grid.dimensions.reduce((a, b) => a * b, 1);
  if (faces.some(words => words.length !== count || !words.every(Number.isFinite))) throw new Error("Invalid MAC face coefficients");
  return count;
}
function inverseFilter(n: number): Float64Array[] {
  const rows = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(2 * n); row[n + i] = 1;
    // += is essential when the periodic domain is smaller than the stencil.
    for (let offset = -2; offset <= 2; offset++) row[wrap(i + offset, n)]! += FACE_FILTER[offset + 2]!;
    return row;
  });
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(rows[row]![col]!) > Math.abs(rows[pivot]![col]!)) pivot = row;
    [rows[col], rows[pivot]] = [rows[pivot]!, rows[col]!];
    const divisor = rows[col]![col]!;
    if (Math.abs(divisor) < 1e-12) throw new Error("Singular periodic face filter");
    for (let at = 0; at < 2 * n; at++) rows[col]![at]! /= divisor;
    for (let row = 0; row < n; row++) if (row !== col) {
      const factor = rows[row]![col]!;
      for (let at = 0; at < 2 * n; at++) rows[row]![at]! -= factor * rows[col]![at]!;
    }
  }
  return rows.map(row => row.slice(n));
}
function applyAxis(grid: PeriodicMacGrid, words: Float64Array, axis: number, matrix: Float64Array[]): Float64Array {
  const result = new Float64Array(words.length), n = grid.dimensions;
  for (let k = 0; k < n[2]; k++) for (let j = 0; j < n[1]; j++) for (let i = 0; i < n[0]; i++) {
    const q = [i, j, k], row = matrix[q[axis]!]!;
    let value = 0;
    for (let source = 0; source < n[axis]!; source++) {
      const at = [...q]; at[axis] = source;
      value += row[source]! * words[macIndex(n, at[0]!, at[1]!, at[2]!)]!;
    }
    result[macIndex(n, i, j, k)] = value;
  }
  return result;
}
export function discreteMacDivergence(grid: PeriodicMacGrid, faces: MacWords): Float64Array {
  const result = new Float64Array(validate(grid, faces)), n = grid.dimensions;
  for (let k = 0; k < n[2]; k++) for (let j = 0; j < n[1]; j++) for (let i = 0; i < n[0]; i++) {
    const id = macIndex(n, i, j, k);
    result[id] = (faces[0][id]! - faces[0][macIndex(n, i - 1, j, k)]!
      + faces[1][id]! - faces[1][macIndex(n, i, j - 1, k)]!
      + faces[2][id]! - faces[2][macIndex(n, i, j, k - 1)]!) / grid.h;
  }
  return result;
}

export class PeriodicC2MacOracle {
  private constructor(readonly grid: PeriodicMacGrid, readonly coefficients: MacWords) {}
  static fromSplineCoefficients(grid: PeriodicMacGrid, coefficients: MacWords): PeriodicC2MacOracle {
    validate(grid, coefficients);
    return new PeriodicC2MacOracle(grid, coefficients.map(v => v.slice()) as unknown as MacWords);
  }
  static fromFaceAverages(grid: PeriodicMacGrid, faces: MacWords): PeriodicC2MacOracle {
    validate(grid, faces);
    let coefficients: Float64Array[] = faces.map(v => v.slice());
    for (let axis = 0; axis < 3; axis++) {
      const inverse = inverseFilter(grid.dimensions[axis]!);
      coefficients = coefficients.map(words => applyAxis(grid, words, axis, inverse));
    }
    return new PeriodicC2MacOracle(grid, coefficients as unknown as MacWords);
  }
  component(point: Triple, component: number, derivative: Triple = [0, 0, 0]): number {
    if (!point.every(Number.isFinite) || !Number.isInteger(component) || component < 0 || component > 2
      || !derivative.every(v => Number.isInteger(v) && v >= 0) || derivative.reduce((a, b) => a + b) > 2) {
      throw new Error("Invalid C2 velocity query");
    }
    const n = this.grid.dimensions;
    const basis = [0, 1, 2].map(axis => {
      const degree = axis === component ? 4 : 3;
      const s = (point[axis]! - this.grid.origin[axis]!) / this.grid.h - (axis === component ? 1 : .5);
      const center = wrap(s, n[axis]!), radius = (degree + 1) / 2;
      const values: { id: number; value: number }[] = [];
      for (let i = Math.floor(center - radius) + 1; i < Math.ceil(center + radius); i++) {
        values.push({ id: i, value: splineDerivative(degree, center - i, derivative[axis]!) / this.grid.h ** derivative[axis]! });
      }
      return values;
    });
    let value = 0;
    for (const z of basis[2]!) for (const y of basis[1]!) for (const x of basis[0]!) {
      value += x.value * y.value * z.value * this.coefficients[component]![macIndex(n, x.id, y.id, z.id)]!;
    }
    return value;
  }
  divergence(point: Triple): number {
    return this.component(point, 0, [1, 0, 0]) + this.component(point, 1, [0, 1, 0]) + this.component(point, 2, [0, 0, 1]);
  }
}
