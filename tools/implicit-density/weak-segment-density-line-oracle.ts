/**
 * Bounded CPU mathematical oracle, not a production transport path.
 *
 * The current periodic C2 cubic potential defines q=clamp(.5-psi/w,0,1).
 * A coefficient segment is advanced by the intrinsic weak PDE equation
 *   integral B_i(q1-q0) - dt*u*integral B_i' average_s(q(psi_s)) = 0.
 * No prescribed destination moments, density clipping after a fit, seed query,
 * native mean correction, or mesh operation occurs here.
 *
 * Scope: 8..16 periodic coefficients, uniform velocity, displacement <=h/2,
 * positive-definite ramp Gram matrix. Exact threshold plateaus and general
 * saturated nullspaces are rejected. Quadrature receipts are numerical error
 * estimates, not outward-rounded integral certificates.
 */
export type WeakDensityLine = Readonly<{
  coefficients: Float64Array;
  length: number;
  width: number;
}>;

export type WeakSegmentOptions = Readonly<{
  maxNewtonIterations?: number;
  maxLineSearchIterations?: number;
  residualTolerance?: number;
  quadratureTolerance?: number;
  maxQuadratureDepth?: number;
  maxQuadratureEvaluations?: number;
  maxGramCondition?: number;
  maxLinearCondition?: number;
}>;

export type WeakLineIntegral = Readonly<{
  amount: number;
  squaredAmount: number;
  gramCondition: number;
  minimumCholeskyPivot: number;
  estimatedAbsoluteError: number;
  evaluations: number;
  pieces: number;
  maximumDepth: number;
}>;

export type WeakSegmentReceipt = Readonly<{
  iterations: number;
  residual: number;
  initial: WeakLineIntegral;
  final: WeakLineIntegral;
  maximumLinearCondition: number;
  totalQuadratureEvaluations: number;
  amountChange: number;
  squaredAmountChange: number;
}>;

type Settings = Required<WeakSegmentOptions>;
type Polynomial = readonly [number, number, number, number];
type Assembly = {
  residual: Float64Array;
  jacobian: Float64Array;
  gram: Float64Array;
  amount: number;
  squaredAmount: number;
  estimatedAbsoluteError: number;
  evaluations: number;
  pieces: number;
  maximumDepth: number;
};
const G4X = [-.8611363115940526, -.3399810435848563, .3399810435848563, .8611363115940526];
const G4W = [.34785484513745385, .6521451548625461, .6521451548625461, .34785484513745385];
const G8X = [-.9602898564975363, -.7966664774136267, -.525532409916329, -.1834346424956498,
  .1834346424956498, .525532409916329, .7966664774136267, .9602898564975363];
const G8W = [.10122853629037626, .22238103445337448, .31370664587788727, .362683783378362,
  .362683783378362, .31370664587788727, .22238103445337448, .10122853629037626];
const clamp = (value: number) => Math.max(0, Math.min(1, value));
const norm = (values: ArrayLike<number>) => {
  let result = 0;
  for (let i = 0; i < values.length; i++) result = Math.max(result, Math.abs(values[i]!));
  return result;
};

function settings(line: WeakDensityLine, options: WeakSegmentOptions): Settings {
  const n = line.coefficients.length;
  if (!Number.isInteger(n) || n < 8 || n > 16 || !Number.isFinite(line.length) || line.length <= 0
    || !Number.isFinite(line.width) || line.width <= 0 || !line.coefficients.every(Number.isFinite))
    throw new Error("unsupported line: finite coefficients, positive length/width and 8..16 sites required");
  const s: Settings = {
    maxNewtonIterations: options.maxNewtonIterations ?? 8,
    maxLineSearchIterations: options.maxLineSearchIterations ?? 8,
    residualTolerance: options.residualTolerance ?? 1e-12 * line.length,
    quadratureTolerance: options.quadratureTolerance ?? 2e-13 * line.length,
    maxQuadratureDepth: options.maxQuadratureDepth ?? 10,
    maxQuadratureEvaluations: options.maxQuadratureEvaluations ?? 100_000,
    maxGramCondition: options.maxGramCondition ?? 1e6,
    maxLinearCondition: options.maxLinearCondition ?? 1e8,
  };
  for (const name of ["maxNewtonIterations", "maxLineSearchIterations", "maxQuadratureDepth", "maxQuadratureEvaluations"] as const)
    if (!Number.isInteger(s[name]) || s[name] < 0) throw new Error(`invalid ${name}`);
  for (const name of ["residualTolerance", "quadratureTolerance", "maxGramCondition", "maxLinearCondition"] as const)
    if (!Number.isFinite(s[name]) || s[name] <= 0) throw new Error(`invalid ${name}`);
  return s;
}

function localPolynomial(coefficients: Float64Array, cell: number): Polynomial {
  const n = coefficients.length;
  const a = coefficients[(cell + n - 1) % n]!, b = coefficients[cell]!,
    c = coefficients[(cell + 1) % n]!, d = coefficients[(cell + 2) % n]!;
  return [(a + 4 * b + c) / 6, (c - a) / 2, (a - 2 * b + c) / 2, (-a + 3 * b - 3 * c + d) / 6];
}
const evaluatePolynomial = (p: readonly number[], t: number): number => {
  let result = 0;
  for (let i = p.length - 1; i >= 0; i--) result = result * t + p[i]!;
  return result;
};

function rejectThresholdPlateau(line: WeakDensityLine): void {
  const n = line.coefficients.length;
  for (let cell = 0; cell < n; cell++) for (const threshold of [-line.width / 2, line.width / 2]) {
    if ([-1, 0, 1, 2].every(offset => line.coefficients[(cell + offset + n) % n] === threshold))
      throw new Error(`unsupported exact saturation-threshold plateau in cell ${cell}`);
  }
}

/** Recursive derivative isolation for degree<=3; all cell roots are considered.
 * Bisection is bounded. Tangencies are retained as cuts, never called dry merely
 * because a fixed quadrature stencil missed a narrow ramp interval. */
function rootsInUnitInterval(input: readonly number[]): number[] {
  const p = [...input];
  while (p.length > 1 && p[p.length - 1] === 0) p.pop();
  if (p.length === 1) return [];
  const critical = rootsInUnitInterval(p.slice(1).map((value, i) => value * (i + 1)));
  const boundaries = [0, ...critical.filter(x => x > 0 && x < 1), 1];
  const roots: number[] = [];
  const scale = Math.max(...p.map(Math.abs));
  for (const x of boundaries) if (Math.abs(evaluatePolynomial(p, x)) <= 32 * Number.EPSILON * scale) roots.push(x);
  for (let i = 0; i + 1 < boundaries.length; i++) {
    let a = boundaries[i]!, b = boundaries[i + 1]!;
    let fa = evaluatePolynomial(p, a), fb = evaluatePolynomial(p, b);
    if (!(fa * fb < 0)) continue;
    for (let iteration = 0; iteration < 54; iteration++) {
      const middle = (a + b) / 2, fm = evaluatePolynomial(p, middle);
      if (middle === a || middle === b || fm === 0) { a = middle; b = middle; break; }
      if ((fa < 0) === (fm < 0)) { a = middle; fa = fm; } else { b = middle; fb = fm; }
    }
    roots.push((a + b) / 2);
  }
  return [...new Set(roots)].sort((a, b) => a - b);
}

/** Exact (up to floating arithmetic) segment integral and its endpoint derivative.
 * derivative wrt endpoint psi1 is -rampFirstMoment/width. */
export function averageClampedSegment(psi0: number, psi1: number, width: number): {
  average: number; rampFirstMoment: number;
} {
  const a = .5 - psi0 / width, b = .5 - psi1 / width;
  if (a === b) return { average: clamp(a), rampFirstMoment: a > 0 && a < 1 ? .5 : 0 };
  if (a <= 0 && b <= 0) return { average: 0, rampFirstMoment: 0 };
  if (a >= 1 && b >= 1) return { average: 1, rampFirstMoment: 0 };
  if (a >= 0 && a <= 1 && b >= 0 && b <= 1)
    return { average: (a + b) / 2, rampFirstMoment: .5 };
  const crossing0 = -a / (b - a), crossing1 = (1 - a) / (b - a);
  const lower = Math.max(0, Math.min(crossing0, crossing1));
  const upper = Math.min(1, Math.max(crossing0, crossing1));
  const cuts = [0, lower, upper, 1];
  let average = 0;
  for (let i = 0; i + 1 < cuts.length; i++) {
    const x = cuts[i]!, y = cuts[i + 1]!;
    average += (y - x) * (clamp(a + x * (b - a)) + clamp(a + y * (b - a))) / 2;
  }
  return { average, rampFirstMoment: (upper - lower) * (upper + lower) / 2 };
}

export function sampleWeakDensityLine(line: WeakDensityLine, x: number): { psi: number; derivative: number; q: number } {
  if (!Number.isFinite(x)) throw new Error("nonfinite sample coordinate");
  const n = line.coefficients.length, h = line.length / n;
  const coordinate = ((x / h) % n + n) % n, cell = Math.floor(coordinate), t = coordinate - cell;
  const p = localPolynomial(line.coefficients, cell), psi = evaluatePolynomial(p, t);
  return { psi, derivative: (p[1] + t * (2 * p[2] + 3 * t * p[3])) / h, q: clamp(.5 - psi / line.width) };
}

function assemble(old: WeakDensityLine, next: Float64Array, velocity: number, dt: number, s: Settings): Assembly {
  rejectThresholdPlateau({ ...old, coefficients: next });
  const n = next.length, h = old.length / n, width = old.width;
  const result: Assembly = { residual: new Float64Array(n), jacobian: new Float64Array(n * n), gram: new Float64Array(n * n),
    amount: 0, squaredAmount: 0, estimatedAbsoluteError: 0, evaluations: 0, pieces: 0, maximumDepth: 0 };
  for (let cell = 0; cell < n; cell++) {
    const p0 = localPolynomial(old.coefficients, cell), p1 = localPolynomial(next, cell);
    const allCuts = [0, 1];
    for (const p of [p0, p1]) for (const threshold of [-width / 2, width / 2])
      allCuts.push(...rootsInUnitInterval([p[0] - threshold, p[1], p[2], p[3]]));
    allCuts.push(...rootsInUnitInterval(p1.map((v, i) => v - p0[i]!)));
    const cuts = [...new Set(allCuts)].sort((a, b) => a - b);
    // Local vector: four residuals, 16 Jacobian entries, 16 ramp Gram
    // entries, amount and squared amount. Only four basis functions overlap.
    const valueAt = (t: number): Float64Array => {
      if (++result.evaluations > s.maxQuadratureEvaluations) throw new Error("quadrature evaluation cap exceeded");
      const t2 = t * t, t3 = t2 * t;
      const basis = [(1 - t) ** 3 / 6, (4 - 6 * t2 + 3 * t3) / 6, (1 + 3 * t + 3 * t2 - 3 * t3) / 6, t3 / 6];
      const derivative = [-((1 - t) ** 2) / (2 * h), (-2 * t + 1.5 * t2) / h,
        (.5 + t - 1.5 * t2) / h, t2 / (2 * h)];
      const psi0 = evaluatePolynomial(p0, t), psi1 = evaluatePolynomial(p1, t);
      const q0 = clamp(.5 - psi0 / width), q1 = clamp(.5 - psi1 / width);
      const { average, rampFirstMoment } = averageClampedSegment(psi0, psi1, width);
      const inRamp1 = psi1 > -width / 2 && psi1 < width / 2 ? 1 : 0;
      const vector = new Float64Array(38);
      for (let i = 0; i < 4; i++) {
        vector[i] = basis[i]! * (q1 - q0) - dt * velocity * derivative[i]! * average;
        for (let j = 0; j < 4; j++) {
          const gram = basis[i]! * basis[j]! * inRamp1;
          vector[4 + 4 * i + j] = (-gram + dt * velocity * derivative[i]! * basis[j]! * rampFirstMoment) / width;
          vector[20 + 4 * i + j] = gram;
        }
      }
      vector[36] = q1; vector[37] = q1 * q1;
      return vector;
    };
    const gauss = (lower: number, upper: number, nodes: readonly number[], weights: readonly number[]) => {
      const sum = new Float64Array(38), middle = (lower + upper) / 2, half = (upper - lower) / 2;
      for (let k = 0; k < nodes.length; k++) {
        const values = valueAt(middle + half * nodes[k]!), weight = half * h * weights[k]!;
        for (let j = 0; j < sum.length; j++) sum[j] = sum[j]! + weight * values[j]!;
      }
      return sum;
    };
    const integrate = (lower: number, upper: number, depth: number): Float64Array => {
      const low = gauss(lower, upper, G4X, G4W), high = gauss(lower, upper, G8X, G8W);
      let error = 0;
      for (let j = 0; j < high.length; j++) error = Math.max(error, Math.abs(high[j]! - low[j]!));
      const tolerance = s.quadratureTolerance * h * (upper - lower) / old.length;
      if (error <= tolerance) {
        result.estimatedAbsoluteError += error;
        result.pieces++;
        result.maximumDepth = Math.max(result.maximumDepth, depth);
        return high;
      }
      if (depth >= s.maxQuadratureDepth) throw new Error(`quadrature refinement rejected: error ${error} > ${tolerance}`);
      const middle = (lower + upper) / 2, a = integrate(lower, middle, depth + 1), b = integrate(middle, upper, depth + 1);
      for (let j = 0; j < a.length; j++) a[j] = a[j]! + b[j]!;
      return a;
    };
    for (let piece = 0; piece + 1 < cuts.length; piece++) {
      if (cuts[piece + 1]! - cuts[piece]! <= 0) continue;
      const vector = integrate(cuts[piece]!, cuts[piece + 1]!, 0);
      for (let i = 0; i < 4; i++) {
        const row = (cell + i + n - 1) % n;
        result.residual[row] = result.residual[row]! + vector[i]!;
        for (let j = 0; j < 4; j++) {
          const index = row * n + (cell + j + n - 1) % n;
          result.jacobian[index] = result.jacobian[index]! + vector[4 + 4 * i + j]!;
          result.gram[index] = result.gram[index]! + vector[20 + 4 * i + j]!;
        }
      }
      result.amount += vector[36]!; result.squaredAmount += vector[37]!;
    }
  }
  return result;
}

/** Tiny dense inverse used only to bound this CPU oracle's solve/conditioning. */
function inverse(matrix: Float64Array, n: number): { inverse: Float64Array; condition: number } {
  const work = matrix.slice(), result = new Float64Array(n * n);
  for (let i = 0; i < n; i++) result[i * n + i] = 1;
  for (let column = 0; column < n; column++) {
    let pivot = column;
    for (let row = column + 1; row < n; row++) if (Math.abs(work[row * n + column]!) > Math.abs(work[pivot * n + column]!)) pivot = row;
    const divisor = work[pivot * n + column]!;
    if (!Number.isFinite(divisor) || divisor === 0) throw new Error("singular matrix rejected");
    for (let j = 0; j < n; j++) {
      const a = column * n + j, b = pivot * n + j;
      [work[a], work[b]] = [work[b]!, work[a]!];
      [result[a], result[b]] = [result[b]!, result[a]!];
      work[a] = work[a]! / divisor; result[a] = result[a]! / divisor;
    }
    for (let row = 0; row < n; row++) if (row !== column) {
      const multiplier = work[row * n + column]!;
      for (let j = 0; j < n; j++) {
        work[row * n + j] = work[row * n + j]! - multiplier * work[column * n + j]!;
        result[row * n + j] = result[row * n + j]! - multiplier * result[column * n + j]!;
      }
    }
  }
  const matrixNorm = (a: Float64Array) => {
    let maximum = 0;
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < n; j++) sum += Math.abs(a[i * n + j]!);
      maximum = Math.max(maximum, sum);
    }
    return maximum;
  };
  const condition = matrixNorm(matrix) * matrixNorm(result);
  if (!Number.isFinite(condition)) throw new Error("nonfinite matrix condition rejected");
  return { inverse: result, condition };
}

function integralReceipt(assembly: Assembly, n: number, s: Settings): WeakLineIntegral {
  const lower = new Float64Array(n * n);
  let minimumCholeskyPivot = Infinity;
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let value = assembly.gram[i * n + j]!;
    for (let k = 0; k < j; k++) value -= lower[i * n + k]! * lower[j * n + k]!;
    if (i === j) {
      if (!(value > 0)) throw new Error("ramp Gram matrix is not positive definite");
      minimumCholeskyPivot = Math.min(minimumCholeskyPivot, value);
      lower[i * n + j] = Math.sqrt(value);
    } else lower[i * n + j] = value / lower[j * n + j]!;
  }
  const gramCondition = inverse(assembly.gram, n).condition;
  if (gramCondition > s.maxGramCondition) throw new Error(`ramp Gram condition rejected: ${gramCondition}`);
  return { amount: assembly.amount, squaredAmount: assembly.squaredAmount, gramCondition, minimumCholeskyPivot,
    estimatedAbsoluteError: assembly.estimatedAbsoluteError, evaluations: assembly.evaluations,
    pieces: assembly.pieces, maximumDepth: assembly.maximumDepth };
}

export function integrateWeakDensityLine(line: WeakDensityLine, options: WeakSegmentOptions = {}): WeakLineIntegral {
  const s = settings(line, options);
  const snapshot = { ...line, coefficients: line.coefficients.slice() };
  const assembled = assemble(snapshot, snapshot.coefficients, 0, 0, s);
  return integralReceipt(assembled, snapshot.coefficients.length, s);
}

export function advanceWeakDensityLine(line: WeakDensityLine, velocity: number, dt: number,
  options: WeakSegmentOptions = {}): { line: WeakDensityLine; receipt: WeakSegmentReceipt } {
  const s = settings(line, options), n = line.coefficients.length;
  if (!Number.isFinite(velocity) || !Number.isFinite(dt) || Math.abs(velocity * dt) > line.length / n / 2)
    throw new Error("unsupported displacement: require finite uniform velocity and at most half a cell");
  const source: WeakDensityLine = { ...line, coefficients: line.coefficients.slice() };
  const initialAssembly = assemble(source, source.coefficients, 0, 0, s), initial = integralReceipt(initialAssembly, n, s);
  let current = source.coefficients.slice(), assembly = assemble(source, current, velocity, dt, s), iterations = 0;
  let totalQuadratureEvaluations = initialAssembly.evaluations + assembly.evaluations, maximumLinearCondition = 0;
  while (norm(assembly.residual) > s.residualTolerance) {
    if (iterations >= s.maxNewtonIterations) throw new Error(`Newton iteration cap rejected: residual ${norm(assembly.residual)}`);
    const solved = inverse(assembly.jacobian, n);
    maximumLinearCondition = Math.max(maximumLinearCondition, solved.condition);
    if (solved.condition > s.maxLinearCondition) throw new Error(`Newton condition rejected: ${solved.condition}`);
    const delta = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) delta[i] = delta[i]! - solved.inverse[i * n + j]! * assembly.residual[j]!;
    let accepted = false;
    for (let backtrack = 0; backtrack <= s.maxLineSearchIterations; backtrack++) {
      const factor = 2 ** -backtrack, candidate = current.map((value, i) => value + factor * delta[i]!);
      if (!candidate.every(Number.isFinite)) continue;
      const candidateAssembly = assemble(source, candidate, velocity, dt, s);
      totalQuadratureEvaluations += candidateAssembly.evaluations;
      if (norm(candidateAssembly.residual) <= (1 - 1e-4 * factor) * norm(assembly.residual)) {
        integralReceipt(candidateAssembly, n, s); // admission before retaining this Newton iterate
        current = candidate; assembly = candidateAssembly; accepted = true; break;
      }
    }
    if (!accepted) throw new Error("Newton line search rejected");
    iterations++;
  }
  const final = integralReceipt(assembly, n, s);
  return { line: { ...source, coefficients: current }, receipt: { iterations, residual: norm(assembly.residual), initial, final,
    maximumLinearCondition, totalQuadratureEvaluations, amountChange: final.amount - initial.amount,
    squaredAmountChange: final.squaredAmount - initial.squaredAmount } };
}
