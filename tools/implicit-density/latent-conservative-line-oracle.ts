/** Research only: one CURRENT C1 latent potential defines q=clamp(.5-psi/w).
 * The clamp is the density's definition, not a repair of an already matched
 * density polynomial. Shared endpoint jets plus a quartic bubble match its
 * own transported density integral. Pure-cell mass cannot determine latent
 * shape, so the transported current midpoint supplies a preferred bubble.
 * Float64 root isolation/integration is numerical, not interval certification.
 */
export type Polynomial = readonly number[];
export interface LatentLine {
  readonly lower: number;
  readonly h: number;
  readonly width: number;
  readonly periodic: boolean;
  readonly value: Float64Array;
  readonly derivative: Float64Array;
  readonly bubble: Float64Array;
  readonly amount: Float64Array;
}
export interface CellSolve {
  polynomial: number[]; delta: number; mean: number; meanResidual: number;
  iterations: number; preferredDelta: number;
}
export interface TransportOptions {
  firstCell?: number; lastCellExclusive?: number;
  maximumPotentialResidual: number;
  maximumDerivativeResidual?: number;
}
export interface LatentTransport {
  field: LatentLine; maximumPotentialResidual: number; maximumDerivativeResidual: number;
  maximumMeanResidual: number; maximumSolveIterations: number;
}
const EPS = Number.EPSILON;
export const latentBeta = (t: number) => 30 * t * t * (1 - t) ** 2;
export const definedDensity = (psi: number, width: number) => Math.max(0, Math.min(1, .5 - psi / width));
export function polynomialValue(p: Polynomial, x: number): number {
  let value = 0; for (let i = p.length - 1; i >= 0; i--) value = value * x + p[i]!; return value;
}
function derivative(p: Polynomial): number[] { return p.slice(1).map((c, i) => (i + 1) * c); }
function primitive(p: Polynomial, x: number): number {
  let value = 0; for (let i = p.length - 1; i >= 0; i--) value = value * x + p[i]! / (i + 1); return value * x;
}
/** Recursive derivative isolation partitions a quartic into monotone pieces;
 * bisection then isolates crossings, including derivative-root tangencies.
 * Tolerances account for float64 evaluation, not a certified root enclosure. */
export function polynomialRoots(p0: Polynomial, lower = 0, upper = 1): number[] {
  const p = [...p0]; while (p.length > 1 && p[p.length - 1] === 0) p.pop();
  if (p.length === 1) return [];
  if (p.length === 2) {
    const root = -p[0]! / p[1]!; return root >= lower && root <= upper ? [root] : [];
  }
  const cuts = [lower, ...polynomialRoots(derivative(p), lower, upper).filter(x => x > lower && x < upper), upper];
  const scale = p.reduce((sum, c) => sum + Math.abs(c) * Math.max(1, Math.abs(lower), Math.abs(upper)) ** (p.length - 1), 0);
  const valueTolerance = 32 * EPS * Math.max(1, scale), roots: number[] = [];
  for (let i = 0; i < cuts.length; i++) {
    const point = cuts[i]!, value = polynomialValue(p, point);
    if (Math.abs(value) <= valueTolerance) roots.push(point);
    if (i + 1 === cuts.length) continue;
    let a = point, b = cuts[i + 1]!, fa = value, fb = polynomialValue(p, b);
    if (!(fa * fb < 0)) continue;
    for (let iteration = 0; iteration < 80 && b - a > 8 * EPS * Math.max(1, Math.abs(a), Math.abs(b)); iteration++) {
      const middle = (a + b) / 2, fm = polynomialValue(p, middle);
      if (fm === 0) { a = middle; b = middle; break; }
      if ((fa < 0) === (fm < 0)) { a = middle; fa = fm; } else { b = middle; fb = fm; }
    }
    roots.push((a + b) / 2);
  }
  return roots.sort((a, b) => a - b).filter((x, i, values) => i === 0 || x - values[i - 1]! > 32 * EPS * Math.max(1, Math.abs(x)));
}
export function polynomialRange(p: Polynomial, lower = 0, upper = 1): readonly [number, number] {
  const values = [lower, upper, ...polynomialRoots(derivative(p), lower, upper)].map(x => polynomialValue(p, x));
  return [Math.min(...values), Math.max(...values)];
}
/** Analytic primitive on each interval cut at psi=+/-width/2 roots. */
export function integrateDefinedDensity(p: Polynomial, width: number, lower = 0, upper = 1): number {
  if (!(width > 0 && Number.isFinite(width) && p.every(Number.isFinite) && upper >= lower)) throw new Error("invalid latent integral");
  const shifted = (level: number) => p.map((v, i) => v - Number(i === 0) * level);
  const cuts = [lower, ...polynomialRoots(shifted(width / 2), lower, upper),
    ...polynomialRoots(shifted(-width / 2), lower, upper), upper].sort((a, b) => a - b);
  let result = 0;
  for (let i = 0; i + 1 < cuts.length; i++) {
    const a = cuts[i]!, b = cuts[i + 1]!; if (b <= a) continue;
    const psi = polynomialValue(p, (a + b) / 2);
    if (psi <= -width / 2) result += b - a;
    else if (psi < width / 2) result += .5 * (b - a) - (primitive(p, b) - primitive(p, a)) / width;
  }
  return result;
}
function cubic(v0: number, d0: number, v1: number, d1: number): number[] {
  return [v0, d0, 3 * (v1 - v0) - 2 * d0 - d1, 2 * (v0 - v1) + d0 + d1, 0];
}
function withBubble(base: Polynomial, delta: number): number[] {
  return [base[0]!, base[1]!, base[2]! + 30 * delta, base[3]! - 60 * delta, 30 * delta];
}
/** The scalar constraint is continuous and decreasing in delta. Strictly
 * partial means have a unique finite solution. Pure means may be impossible
 * for the fixed endpoint jets, and otherwise leave latent shape undetermined. */
export function solveLatentCell(v0: number, d0: number, v1: number, d1: number, target: number,
  width: number, preferredDelta = 0, tolerance = 2e-13): CellSolve {
  if (![v0, d0, v1, d1, target, width, preferredDelta, tolerance].every(Number.isFinite)
    || !(width > 0 && target >= 0 && target <= 1 && tolerance > 0)) throw new Error("invalid nonlinear latent constraint");
  const base = cubic(v0, d0, v1, d1);
  if (target === 0 && (v0 < width / 2 || v1 < width / 2
    || v0 === width / 2 && d0 < 0 || v1 === width / 2 && d1 > 0)) throw new Error("pure dry target infeasible for fixed endpoint jets");
  if (target === 1 && (v0 > -width / 2 || v1 > -width / 2
    || v0 === -width / 2 && d0 > 0 || v1 === -width / 2 && d1 < 0)) throw new Error("pure full target infeasible for fixed endpoint jets");
  let evaluations = 0;
  const evaluate = (delta: number): CellSolve => {
    evaluations++;
    const polynomial = withBubble(base, delta), mean = integrateDefinedDensity(polynomial, width);
    return { polynomial, delta, mean, meanResidual: mean - target, iterations: evaluations, preferredDelta };
  };
  const acceptable = (value: CellSolve) => target === 0 || target === 1 ? value.mean === target : Math.abs(value.meanResidual) <= tolerance;
  let preferred = evaluate(preferredDelta);
  if (acceptable(preferred)) return preferred;
  let lower = preferredDelta, upper = preferredDelta, left = preferred, right = preferred, radius = width;
  for (let iteration = 0; iteration < 60 && !(left.mean >= target && right.mean <= target); iteration++) {
    if (left.mean < target) { lower = preferredDelta - radius; left = evaluate(lower); }
    if (right.mean > target) { upper = preferredDelta + radius; right = evaluate(upper); }
    radius *= 2;
  }
  if (!(left.mean >= target && right.mean <= target)) throw new Error("latent mass solve exceeded finite bracket budget");
  if (target === 0 || target === 1) {
    const pure = target === 0 ? right : left;
    if (!acceptable(pure)) throw new Error("pure target integral is numerically unresolved");
    return pure;
  }
  let best = Math.abs(left.meanResidual) < Math.abs(right.meanResidual) ? left : right;
  for (let iteration = 0; iteration < 100; iteration++) {
    const mid = evaluate((lower + upper) / 2);
    if (Math.abs(mid.meanResidual) < Math.abs(best.meanResidual)) best = mid;
    if (acceptable(mid)) return mid;
    if (mid.mean > target) lower = mid.delta; else upper = mid.delta;
    if (upper - lower < 8 * EPS * Math.max(1, Math.abs(lower), Math.abs(upper))) break;
  }
  if (!acceptable(best)) throw new Error(`latent mass solve residual ${best.meanResidual} exceeds ${tolerance}`);
  return best;
}
export function latentCell(line: LatentLine, index: number): number[] {
  const n = line.bubble.length, i = line.periodic ? ((index % n) + n) % n : index;
  if (i < 0 || i >= n) throw new Error("missing current latent support");
  return withBubble(cubic(line.value[i]!, line.h * line.derivative[i]!, line.value[i + 1]!, line.h * line.derivative[i + 1]!), line.bubble[i]!);
}
function latticeSample(line: LatentLine, coordinate: number): readonly [number, number] {
  const n = line.bubble.length;
  let x = coordinate;
  if (line.periodic) x = ((x % n) + n) % n;
  if (x < 0 || x > n) throw new Error("missing current latent support");
  const cell = Math.min(n - 1, Math.floor(x)), local = x - cell, p = latentCell(line, cell);
  return [polynomialValue(p, local), polynomialValue(derivative(p), local) / line.h];
}
export function sampleLatentLine(line: LatentLine, x: number): readonly [number, number, number] {
  const [psi, slope] = latticeSample(line, (x - line.lower) / line.h);
  return [psi, slope, definedDensity(psi, line.width)];
}
function integrateLattice(line: LatentLine, lower: number, upper: number, exactFullAmount = upper - lower): number {
  if (!line.periodic && (lower < 0 || upper > line.bubble.length)) throw new Error("missing current latent integration support");
  let amount = 0, allDry = true, allFull = true;
  for (let cell = Math.floor(lower); cell < Math.ceil(upper); cell++) {
    const a = Math.max(lower, cell) - cell, b = Math.min(upper, cell + 1) - cell;
    if (!(b > a)) continue;
    const p = latentCell(line, cell), range = polynomialRange(p, a, b);
    allDry &&= range[0] >= line.width / 2; allFull &&= range[1] <= -line.width / 2;
    amount += integrateDefinedDensity(p, line.width, a, b);
  }
  // A certified-in-real-arithmetic full interval has its geometric measure.
  // A transported unit support supplies 1 explicitly, avoiding subtraction
  // roundoff in (departure+1)-departure. This does not clamp a mixed integral.
  return allDry ? 0 : allFull ? exactFullAmount : amount;
}
export function integrateLatentLine(line: LatentLine, lower: number, upper: number): number {
  if (!(upper >= lower && Number.isFinite(lower) && Number.isFinite(upper))) throw new Error("invalid forward integral");
  return line.h * integrateLattice(line, (lower - line.lower) / line.h, (upper - line.lower) / line.h);
}
export function initializeLatentLine(cells: number, lower: number, upper: number, width: number,
  potential: (x: number) => number, slope: (x: number) => number,
  densityIntegral: (lower: number, upper: number) => number, periodic = false): LatentLine {
  if (!(Number.isSafeInteger(cells) && cells > 1 && upper > lower && width > 0)) throw new Error("invalid latent line");
  const h = (upper - lower) / cells;
  const field: LatentLine = { lower, h, width, periodic,
    value: Float64Array.from({ length: cells + 1 }, (_, i) => potential(lower + i * h)),
    derivative: Float64Array.from({ length: cells + 1 }, (_, i) => slope(lower + i * h)),
    bubble: new Float64Array(cells), amount: new Float64Array(cells) };
  if (periodic && (Math.abs(field.value[0]! - field.value[cells]!) > 1e-13
    || Math.abs(field.derivative[0]! - field.derivative[cells]!) > 1e-12)) throw new Error("periodic endpoint jets disagree");
  if (periodic) { field.value[cells] = field.value[0]!; field.derivative[cells] = field.derivative[0]!; }
  for (let i = 0; i < cells; i++) {
    const a = lower + i * h, b = lower + (i + 1) * h;
    const H = cubic(field.value[i]!, h * field.derivative[i]!, field.value[i + 1]!, h * field.derivative[i + 1]!);
    const preferred = (potential((a + b) / 2) - polynomialValue(H, .5)) / latentBeta(.5);
    const solution = solveLatentCell(field.value[i]!, h * field.derivative[i]!, field.value[i + 1]!, h * field.derivative[i + 1]!, densityIntegral(a, b) / h, width, preferred);
    field.bubble[i] = solution.delta; field.amount[i] = solution.mean * h;
  }
  return field;
}
function composeShift(p: Polynomial, offset: number): number[] {
  const out = [0, 0, 0, 0, 0];
  const binomial = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]];
  for (let i = 0; i < p.length; i++) for (let j = 0; j <= i; j++) out[j]! += p[i]! * binomial[i]![j]! * offset ** (i - j);
  return out;
}
export function transportLatentLine(line: LatentLine, displacement: number, options: TransportOptions): LatentTransport {
  const start = options.firstCell ?? 0, end = options.lastCellExclusive ?? line.bubble.length, n = end - start;
  if (!(Number.isFinite(displacement) && Number.isSafeInteger(start) && Number.isSafeInteger(end) && n > 1 && start >= 0 && end <= line.bubble.length
    && options.maximumPotentialResidual >= 0 && (options.maximumDerivativeResidual ?? Infinity) >= 0)) throw new Error("invalid latent transport contract");
  if (line.periodic && (start !== 0 || end !== line.bubble.length)) throw new Error("periodic cropping is unsupported");
  const field: LatentLine = { lower: line.lower + start * line.h, h: line.h, width: line.width, periodic: line.periodic,
    value: new Float64Array(n + 1), derivative: new Float64Array(n + 1), bubble: new Float64Array(n), amount: new Float64Array(n) };
  const shift = displacement / line.h;
  for (let i = 0; i <= n; i++) [field.value[i], field.derivative[i]] = latticeSample(line, start + i - shift);
  if (line.periodic) { field.value[n] = field.value[0]!; field.derivative[n] = field.derivative[0]!; }
  let maximumPotentialResidual = 0, maximumDerivativeResidual = 0, maximumMeanResidual = 0, maximumSolveIterations = 0;
  for (let i = 0; i < n; i++) {
    const departure = start + i - shift;
    const H = cubic(field.value[i]!, line.h * field.derivative[i]!, field.value[i + 1]!, line.h * field.derivative[i + 1]!);
    const preferred = (latticeSample(line, departure + .5)[0] - polynomialValue(H, .5)) / latentBeta(.5);
    const target = integrateLattice(line, departure, departure + 1, 1);
    const solution = solveLatentCell(field.value[i]!, line.h * field.derivative[i]!, field.value[i + 1]!, line.h * field.derivative[i + 1]!, target, line.width, preferred);
    field.bubble[i] = solution.delta; field.amount[i] = solution.mean * line.h;
    maximumMeanResidual = Math.max(maximumMeanResidual, Math.abs(solution.meanResidual)); maximumSolveIterations = Math.max(maximumSolveIterations, solution.iterations);
    // Compare complete polynomial pieces, not a finite probe-grid residual.
    for (let source = Math.floor(departure); source < Math.ceil(departure + 1); source++) {
      const a = Math.max(0, source - departure), b = Math.min(1, source + 1 - departure);
      if (!(b > a)) continue;
      const old = composeShift(latentCell(line, source), departure - source);
      const difference = solution.polynomial.map((v, k) => v - old[k]!);
      const range = polynomialRange(difference, a, b), slopeRange = polynomialRange(derivative(difference), a, b);
      maximumPotentialResidual = Math.max(maximumPotentialResidual, Math.abs(range[0]), Math.abs(range[1]));
      maximumDerivativeResidual = Math.max(maximumDerivativeResidual, Math.abs(slopeRange[0]) / line.h, Math.abs(slopeRange[1]) / line.h);
    }
  }
  if (maximumPotentialResidual > options.maximumPotentialResidual || maximumDerivativeResidual > (options.maximumDerivativeResidual ?? Infinity)) {
    throw new Error(`latent transport shape residual rejected: potential=${maximumPotentialResidual}, derivative=${maximumDerivativeResidual}`);
  }
  return { field, maximumPotentialResidual, maximumDerivativeResidual, maximumMeanResidual, maximumSolveIterations };
}
