/** Small CPU research oracle, never a production fluid or surface path.
 *
 * One periodic 1D field carries shared point values/derivatives and cell
 * integrals. A quartic satisfies all five constraints. Translation evaluates
 * that same current field and integrates its departure intervals. There is no
 * seed shape, independent mean-interpolation target, mesh, clipping or filter.
 *
 * This isolates the continuity/conservation idea behind CIP-CSL4; it does not
 * implement that paper's complete solver or establish a 3D positivity scheme.
 * https://doi.org/10.1175/1520-0493(2001)129<0332:AECSLS>2.0.CO;2
 */
export interface ConservativeHermiteLine {
  readonly length: number;
  readonly value: Float64Array;
  readonly derivative: Float64Array;
  readonly amount: Float64Array;
}
type Quartic = readonly [number, number, number, number, number];
const mod = (x: number, n: number): number => ((x % n) + n) % n;

export function hermiteCell(line: ConservativeHermiteLine, index: number): Quartic {
  const n = line.value.length, h = line.length / n;
  const i = mod(index, n), j = (i + 1) % n;
  const a = line.value[i]!, b = h * line.derivative[i]!;
  const r = line.value[j]! - a - b, d = h * line.derivative[j]! - b;
  const integral = line.amount[i]! / h - a - b / 2;
  return [a, b, 30 * integral - 12 * r + 1.5 * d,
    28 * r - 4 * d - 60 * integral, 30 * integral - 15 * r + 2.5 * d];
}
export function quarticValue(p: Quartic, x: number): number {
  return p[0] + x * (p[1] + x * (p[2] + x * (p[3] + x * p[4])));
}
export function quarticDerivative(p: Quartic, x: number): number {
  return p[1] + x * (2 * p[2] + x * (3 * p[3] + x * 4 * p[4]));
}
function primitive(p: Quartic, x: number): number {
  return x * (p[0] + x * (p[1] / 2 + x * (p[2] / 3 + x * (p[3] / 4 + x * p[4] / 5))));
}
export function sampleHermiteLine(line: ConservativeHermiteLine, x: number): readonly [number, number] {
  const h = line.length / line.value.length, coordinate = mod(x, line.length) / h;
  const cell = Math.floor(coordinate), local = coordinate - cell;
  const p = hermiteCell(line, cell);
  return [quarticValue(p, local), quarticDerivative(p, local) / h];
}

/** Integrate a forward interval shorter than or equal to the periodic domain.
 * Integer-lattice cuts avoid epsilon stepping and skipped slivers at faces.
 */
export function integrateHermiteLine(line: ConservativeHermiteLine, lower: number, upper: number): number {
  if (!(upper >= lower && upper - lower <= line.length * (1 + 1e-14))) throw new Error("invalid periodic integration interval");
  if (upper === lower) return 0;
  const h = line.length / line.value.length;
  const start = lower / h, end = upper / h;
  let sum = 0;
  for (let cell = Math.floor(start); cell < Math.ceil(end); cell++) {
    const a = Math.max(start, cell) - cell, b = Math.min(end, cell + 1) - cell;
    if (b > a) { const p = hermiteCell(line, cell); sum += h * (primitive(p, b) - primitive(p, a)); }
  }
  return sum;
}

export function translateHermiteLine(line: ConservativeHermiteLine, displacement: number): ConservativeHermiteLine {
  const n = line.value.length, h = line.length / n;
  if (!(Number.isFinite(displacement) && n > 1 && Number.isFinite(line.length) && line.length > 0
    && line.derivative.length === n && line.amount.length === n)) throw new Error("invalid shared Hermite line");
  const next: ConservativeHermiteLine = { length: line.length,
    value: new Float64Array(n), derivative: new Float64Array(n), amount: new Float64Array(n) };
  const shift = mod(displacement, line.length);
  for (let i = 0; i < n; i++) {
    const lower = i * h - shift;
    [next.value[i], next.derivative[i]] = sampleHermiteLine(line, lower);
    next.amount[i] = integrateHermiteLine(line, lower, (i + 1) * h - shift);
  }
  return next;
}

export function initializeHermiteLine(cells: number, length: number,
  value: (x: number) => number, derivative: (x: number) => number,
  integral: (lower: number, upper: number) => number): ConservativeHermiteLine {
  if (!(Number.isSafeInteger(cells) && cells > 1 && Number.isFinite(length) && length > 0)) throw new Error("invalid line dimensions");
  const h = length / cells;
  return { length, value: Float64Array.from({ length: cells }, (_, i) => value(i * h)),
    derivative: Float64Array.from({ length: cells }, (_, i) => derivative(i * h)),
    amount: Float64Array.from({ length: cells }, (_, i) => integral(i * h, (i + 1) * h)) };
}
