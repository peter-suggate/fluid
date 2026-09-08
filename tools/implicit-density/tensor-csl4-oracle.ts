/** Bounded CPU mathematical oracle, never a production fluid implementation.
 *
 * Tensorize the five 1D quartic constraints [V0,hD0,V1,hD1,mean].
 * Each periodic lattice entity stores the Cartesian product of V, hD, mean:
 * 27 scalars per bulk cell, rather than 125 independently stored coefficients.
 * Translation transfers the mixed functionals of the CURRENT polynomial.
 * Neither positivity clipping nor a seed geometry occurs in this module.
 */
export type Triple = readonly [number, number, number];
export interface TensorCSL4Field {
  readonly dimensions: Triple;
  readonly lengths: Triple;
  readonly data: Float64Array;
}
export interface SeparableFactor {
  value(x: number): number;
  derivative(x: number): number;
  integral(lower: number, upper: number): number;
}
export interface SeparableTerm { scale: number; factors: readonly [SeparableFactor, SeparableFactor, SeparableFactor] }
export type AxisFunctional = { kind: "point"; position: number; derivative?: boolean }
  | { kind: "integral"; lower: number; upper: number };
type Row = readonly (readonly [number, number])[];
const mod = (x: number, n: number) => ((x % n) + n) % n;
// Columns are the five cardinal functions, rows their power coefficients.
const POWER = [
  [1, 0, 0, 0, 0], [0, 1, 0, 0, 0],
  [-18, -4.5, -12, 1.5, 30], [32, 6, 28, -4, -60], [-15, -2.5, -15, 2.5, 30],
];
const BERNSTEIN = [
  [1, 0, 0, 0, 0], [1, .25, 0, 0, 0], [-2, -.25, -2, .25, 5],
  [0, 0, 1, -.25, 0], [0, 0, 1, 0, 0],
];
function localSlots(cell: number, n: number): number[] {
  const i = 3 * mod(cell, n), j = 3 * mod(cell + 1, n);
  return [i, i + 1, j, j + 1, i + 2];
}
function basis(t: number, derivative = false): number[] {
  return Array.from({ length: 5 }, (_, j) => derivative
    ? POWER[1]![j]! + t * (2 * POWER[2]![j]! + t * (3 * POWER[3]![j]! + 4 * t * POWER[4]![j]!))
    : POWER[0]![j]! + t * (POWER[1]![j]! + t * (POWER[2]![j]! + t * (POWER[3]![j]! + t * POWER[4]![j]!))));
}
function integralBasis(a: number, b: number): number[] {
  return Array.from({ length: 5 }, (_, j) => {
    let result = 0;
    for (let k = 0; k < 5; k++) result += POWER[k]![j]! * (b ** (k + 1) - a ** (k + 1)) / (k + 1);
    return result;
  });
}
function pointRow(n: number, coordinate: number, derivative = false): Row {
  const cell = Math.floor(coordinate), slots = localSlots(cell, n), weights = basis(coordinate - cell, derivative);
  return slots.map((slot, j) => [slot, weights[j]!] as const);
}
function integralRow(n: number, lower: number, upper: number): Row {
  if (!(Number.isFinite(lower) && Number.isFinite(upper) && upper >= lower && upper - lower <= n * (1 + 1e-14))) {
    throw new Error("integration interval must be finite, forward and no longer than one period");
  }
  const row = new Map<number, number>();
  for (let cell = Math.floor(lower); cell < Math.ceil(upper); cell++) {
    const a = Math.max(lower, cell) - cell, b = Math.min(upper, cell + 1) - cell;
    if (!(b > a)) continue;
    const weights = integralBasis(a, b), slots = localSlots(cell, n);
    for (let j = 0; j < 5; j++) row.set(slots[j]!, (row.get(slots[j]!) ?? 0) + weights[j]!);
  }
  return [...row];
}
function validateShape(dimensions: Triple, lengths: Triple): void {
  if (!dimensions.every(n => Number.isSafeInteger(n) && n > 1)
    || !lengths.every(l => Number.isFinite(l) && l > 0)) throw new Error("invalid tensor dimensions");
}
export function tensorSlot(field: TensorCSL4Field, cells: Triple, types: Triple): number {
  const [nx, ny, nz] = field.dimensions;
  return 3 * mod(cells[0], nx) + types[0] + 3 * nx * (3 * mod(cells[1], ny) + types[1]
    + 3 * ny * (3 * mod(cells[2], nz) + types[2]));
}
/** Analytic separable inputs provide all mixed moments without quadrature. */
export function initializeTensorCSL4(dimensions: Triple, lengths: Triple, terms: readonly SeparableTerm[]): TensorCSL4Field {
  validateShape(dimensions, lengths);
  const size = dimensions.map(n => 3 * n), data = new Float64Array(size[0]! * size[1]! * size[2]!);
  for (const term of terms) {
    const axes = term.factors.map((factor, axis) => {
      const h = lengths[axis]! / dimensions[axis]!;
      return Float64Array.from({ length: size[axis]! }, (_, slot) => {
        const i = Math.floor(slot / 3), kind = slot % 3;
        return kind === 0 ? factor.value(i * h) : kind === 1 ? h * factor.derivative(i * h)
          : factor.integral(i * h, (i + 1) * h) / h;
      });
    });
    let index = 0;
    for (let z = 0; z < size[2]!; z++) for (let y = 0; y < size[1]!; y++) for (let x = 0; x < size[0]!; x++) {
      data[index++]! += term.scale * axes[0]![x]! * axes[1]![y]! * axes[2]![z]!;
    }
  }
  return { dimensions, lengths, data };
}
function contract(field: TensorCSL4Field, rows: readonly [Row, Row, Row]): number {
  const sx = 3 * field.dimensions[0], sy = 3 * field.dimensions[1];
  let result = 0;
  for (const [z, wz] of rows[2]) for (const [y, wy] of rows[1]) for (const [x, wx] of rows[0]) {
    result += wx * wy * wz * field.data[x + sx * (y + sy * z)]!;
  }
  return result;
}
/** Point/mixed derivative/partial box-integral query of one current field.
 * Integral and derivative units are physical, unlike normalized stored DOFs. */
export function tensorFunctional(field: TensorCSL4Field, functionals: readonly [AxisFunctional, AxisFunctional, AxisFunctional]): number {
  let scale = 1;
  const rows = functionals.map((f, axis) => {
    const n = field.dimensions[axis]!, h = field.lengths[axis]! / n;
    if (f.kind === "integral") { scale *= h; return integralRow(n, f.lower / h, f.upper / h); }
    if (!Number.isFinite(f.position)) throw new Error("nonfinite point");
    if (f.derivative) scale /= h;
    return pointRow(n, mod(f.position, field.lengths[axis]!) / h, f.derivative);
  }) as [Row, Row, Row];
  return scale * contract(field, rows);
}
export function sampleTensorCSL4(field: TensorCSL4Field, point: Triple): readonly [number, number, number, number] {
  const f = point.map(position => ({ kind: "point" as const, position })) as [AxisFunctional, AxisFunctional, AxisFunctional];
  const result = [tensorFunctional(field, f)];
  for (let axis = 0; axis < 3; axis++) {
    const d = [...f] as [AxisFunctional, AxisFunctional, AxisFunctional];
    d[axis] = { kind: "point", position: point[axis]!, derivative: true };
    result.push(tensorFunctional(field, d));
  }
  return result as unknown as readonly [number, number, number, number];
}
export function tensorAmount(field: TensorCSL4Field): number {
  const [nx, ny, nz] = field.dimensions;
  let result = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    result += field.data[tensorSlot(field, [x, y, z], [2, 2, 2])]!;
  }
  return result * field.lengths[0] * field.lengths[1] * field.lengths[2] / (nx * ny * nz);
}
/** Exact tensor-functional measurement of uniform translation. Reconstructing
 * its new quartic is an approximation unless the translated field lies in the
 * target space. This is not a nonlinear dimensional-splitting flow solver. */
export function translateTensorCSL4(field: TensorCSL4Field, displacement: Triple): TensorCSL4Field {
  if (!displacement.every(Number.isFinite)) throw new Error("nonfinite displacement");
  const sizes = field.dimensions.map(n => 3 * n), strides = [1, sizes[0]!, sizes[0]! * sizes[1]!];
  let source = field.data;
  for (let axis = 0; axis < 3; axis++) {
    const n = field.dimensions[axis]!, h = field.lengths[axis]! / n, shift = mod(displacement[axis]!, field.lengths[axis]!) / h;
    if (shift === 0) continue;
    const rows = Array.from({ length: 3 * n }, (_, slot) => {
      const i = Math.floor(slot / 3), kind = slot % 3;
      return kind === 2 ? integralRow(n, i - shift, i + 1 - shift) : pointRow(n, i - shift, kind === 1);
    });
    const next = new Float64Array(source.length), stride = strides[axis]!, block = stride * sizes[axis]!;
    for (let base = 0; base < source.length; base += block) for (let offset = 0; offset < stride; offset++) {
      for (let slot = 0; slot < sizes[axis]!; slot++) {
        let value = 0;
        for (const [old, weight] of rows[slot]!) value += weight * source[base + offset + old * stride]!;
        next[base + offset + slot * stride] = value;
      }
    }
    source = next;
  }
  return { dimensions: field.dimensions, lengths: field.lengths, data: source === field.data ? source.slice() : source };
}
export function tensorCellMoments(field: TensorCSL4Field, cell: Triple): Float64Array {
  const slots = cell.map((i, axis) => localSlots(i, field.dimensions[axis]!)), sx = 3 * field.dimensions[0], sy = 3 * field.dimensions[1];
  const values = new Float64Array(125);
  let index = 0;
  for (const z of slots[2]!) for (const y of slots[1]!) for (const x of slots[0]!) values[index++] = field.data[x + sx * (y + sy * z)]!;
  return values;
}
function tensorTransform(input: Float64Array, matrix: readonly (readonly number[])[]): Float64Array {
  let values = input;
  for (const stride of [1, 5, 25]) {
    const next = new Float64Array(125), block = 5 * stride;
    for (let base = 0; base < 125; base += block) for (let offset = 0; offset < stride; offset++) {
      for (let out = 0; out < 5; out++) for (let old = 0; old < 5; old++) {
        next[base + offset + out * stride]! += matrix[out]![old]! * values[base + offset + old * stride]!;
      }
    }
    values = next;
  }
  return values;
}
export function tensorCellBernstein(field: TensorCSL4Field, cell: Triple): Float64Array {
  return tensorTransform(tensorCellMoments(field, cell), BERNSTEIN);
}
/** A sufficient real-arithmetic certificate, evaluated here in float64. No
 * interval-rounding guarantee is claimed. A failed bound is rejection, not
 * proof that q itself is negative, and no input coefficient is changed. */
export function tensorRangeAdmission(field: TensorCSL4Field): { admitted: boolean; lower: number; upper: number; rejectedCells: number } {
  let lower = Infinity, upper = -Infinity, rejectedCells = 0;
  const [nx, ny, nz] = field.dimensions;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const b = tensorCellBernstein(field, [x, y, z]);
    let valid = true;
    for (const value of b) { lower = Math.min(lower, value); upper = Math.max(upper, value); if (!(value >= 0 && value <= 1)) valid = false; }
    if (!valid) rejectedCells++;
  }
  return { admitted: rejectedCells === 0, lower, upper, rejectedCells };
}
/** Coefficient-level value/normal-derivative trace comparison over all faces.
 * Both normals are measured in the same positive coordinate direction. */
export function tensorFaceIdentity(field: TensorCSL4Field): { valueError: number; normalError: number } {
  let valueError = 0, normalError = 0;
  const [nx, ny, nz] = field.dimensions, strides = [1, 5, 25];
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const cell: [number, number, number] = [x, y, z], a = tensorCellBernstein(field, cell);
    for (let axis = 0; axis < 3; axis++) {
      const neighbor = [...cell] as [number, number, number]; neighbor[axis]!++;
      const b = tensorCellBernstein(field, neighbor), stride = strides[axis]!, h = field.lengths[axis]! / field.dimensions[axis]!;
      for (let i = 0; i < 125; i++) if (Math.floor(i / stride) % 5 === 0) {
        valueError = Math.max(valueError, Math.abs(a[i + 4 * stride]! - b[i]!));
        normalError = Math.max(normalError, Math.abs(4 * (a[i + 4 * stride]! - a[i + 3 * stride]! - b[i + stride]! + b[i]!) / h));
      }
    }
  }
  return { valueError, normalError };
}
