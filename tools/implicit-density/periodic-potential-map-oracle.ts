import { macIndex, splineDerivative, type MacWords, type PeriodicMacGrid, type Triple } from "./periodic-c2-mac-oracle";

/** CPU research only. Synthetic periodic edge-potential spline coefficients
 * define one C2 curl velocity. No inverse-curl compiler, VEX conversion, map
 * compression, GPU or production transport is implemented here.
 * Ax is centered at (i+.5,j+1,k+1), using B3(x)B4(y)B4(z); cyclic for Ay/Az.
 * The discrete curl gives MAC SPLINE COEFFICIENTS, not supplied face averages.
 */
export type Matrix3 = readonly [number, number, number, number, number, number, number, number, number];
export const IDENTITY3: Matrix3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
export interface MapOptions {
  maximumIterations?: number;
  residualTolerance?: number;
  maximumConditionNumber?: number;
  maximumContraction?: number;
}
export interface MapJet {
  point: Triple;
  jacobian: Matrix3;
  hessians: readonly [Matrix3, Matrix3, Matrix3];
  residual: number;
  iterations: number;
  conditionNumber: number;
  contractionBound: number;
  /** Maximum per-pair residual/(1-kappa), not a composed global error bound.
   * This real-arithmetic estimate excludes floating-point evaluation error. */
  maximumPairPositionErrorBound: number;
}
const zeros = (): Matrix3 => [0, 0, 0, 0, 0, 0, 0, 0, 0];
const wrap = (v: number, n: number) => ((v % n) + n) % n;
function asMatrix(v: number[]): Matrix3 { return v as unknown as Matrix3; }
function multiply(a: Matrix3, b: Matrix3): Matrix3 {
  return asMatrix(Array.from({ length: 9 }, (_, i) => {
    const row = Math.floor(i / 3), col = i % 3;
    return a[3 * row]! * b[col]! + a[3 * row + 1]! * b[3 + col]! + a[3 * row + 2]! * b[6 + col]!;
  }));
}
export function determinant3(a: Matrix3): number {
  return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
}
function composeJets(outer: MapJet, inner: MapJet): MapJet {
  const jacobian = multiply(outer.jacobian, inner.jacobian);
  const hessians = [0, 1, 2].map(component => asMatrix(Array.from({ length: 9 }, (_, entry) => {
    const a = Math.floor(entry / 3), b = entry % 3;
    let value = 0;
    for (let i = 0; i < 3; i++) {
      value += outer.jacobian[3 * component + i]! * inner.hessians[i]![entry]!;
      for (let j = 0; j < 3; j++) value += inner.jacobian[3 * i + a]! * outer.hessians[component]![3 * i + j]! * inner.jacobian[3 * j + b]!;
    }
    return value;
  }))) as [Matrix3, Matrix3, Matrix3];
  return { point: outer.point, jacobian, hessians,
    residual: Math.max(outer.residual, inner.residual), iterations: outer.iterations + inner.iterations,
    conditionNumber: Math.max(outer.conditionNumber, inner.conditionNumber), contractionBound: Math.max(outer.contractionBound, inner.contractionBound),
    maximumPairPositionErrorBound: Math.max(outer.maximumPairPositionErrorBound, inner.maximumPairPositionErrorBound) };
}
function identityJet(point: Triple): MapJet {
  return { point, jacobian: IDENTITY3, hessians: [zeros(), zeros(), zeros()], residual: 0, iterations: 0, conditionNumber: 1, contractionBound: 0,
    maximumPairPositionErrorBound: 0 };
}

export class PeriodicPotentialMapOracle {
  private readonly edgeCoefficients: MacWords;
  readonly grid: PeriodicMacGrid;
  readonly mean: Triple;
  private readonly coefficientBounds: Triple;
  constructor(grid: PeriodicMacGrid, potentials: MacWords, mean: Triple = [0, 0, 0]) {
    const count = grid.dimensions.reduce((a, b) => a * b, 1);
    if (!grid.dimensions.every(n => Number.isInteger(n) && n >= 3 && n <= 8)
      || !(Number.isFinite(grid.h) && grid.h > 0) || !grid.origin.every(Number.isFinite)
      || !mean.every(Number.isFinite) || potentials.some(v => v.length !== count || !v.every(Number.isFinite))) throw new Error("invalid bounded edge-potential oracle");
    this.grid = Object.freeze({ dimensions: Object.freeze([...grid.dimensions]) as Triple, h: grid.h, origin: Object.freeze([...grid.origin]) as Triple });
    this.edgeCoefficients = potentials.map(v => v.slice()) as unknown as MacWords;
    this.mean = Object.freeze([...mean]) as Triple;
    this.coefficientBounds = potentials.map(v => Math.max(...v.map(Math.abs))) as unknown as Triple;
  }
  /** QA copies: external writes cannot invalidate the cached global bounds. */
  get potentials(): MacWords { return this.edgeCoefficients.map(v => v.slice()) as unknown as MacWords; }
  potential(point: Triple, component: number, derivative: Triple = [0, 0, 0]): number {
    if (!point.every(Number.isFinite) || !Number.isInteger(component) || component < 0 || component > 2
      || !derivative.every(order => Number.isInteger(order) && order >= 0) || derivative.reduce((a, b) => a + b) > 3) throw new Error("invalid potential query");
    const { dimensions: n, h, origin } = this.grid;
    const axes = [0, 1, 2].map(axis => {
      const degree = axis === component ? 3 : 4, radius = (degree + 1) / 2;
      const x = wrap((point[axis]! - origin[axis]!) / h - (axis === component ? .5 : 1), n[axis]!);
      const result: { index: number; weight: number }[] = [];
      for (let i = Math.floor(x - radius) + 1; i < Math.ceil(x + radius); i++) {
        result.push({ index: i, weight: splineDerivative(degree, x - i, derivative[axis]!) / h ** derivative[axis]! });
      }
      return result;
    });
    let result = 0;
    for (const z of axes[2]!) for (const y of axes[1]!) for (const x of axes[0]!) {
      result += x.weight * y.weight * z.weight * this.edgeCoefficients[component]![macIndex(n, x.index, y.index, z.index)]!;
    }
    return result;
  }
  velocity(point: Triple, derivative: Triple = [0, 0, 0]): Triple {
    if (derivative.reduce((a, b) => a + b) > 2) throw new Error("velocity derivative order exceeds C2 contract");
    return [0, 1, 2].map(component => {
      const a = (component + 1) % 3, b = (component + 2) % 3;
      const da = [...derivative] as [number, number, number], db = [...derivative] as [number, number, number];
      da[a]!++; db[b]!++;
      return this.potential(point, b, da) - this.potential(point, a, db)
        + (derivative.every(v => v === 0) ? this.mean[component]! : 0);
    }) as unknown as Triple;
  }
  curlSplineCoefficients(): MacWords {
    const { dimensions: n, h } = this.grid, count = n[0] * n[1] * n[2];
    const output = [new Float64Array(count), new Float64Array(count), new Float64Array(count)] as const;
    for (let k = 0; k < n[2]; k++) for (let j = 0; j < n[1]; j++) for (let i = 0; i < n[0]; i++) {
      const coordinate = [i, j, k], id = macIndex(n, i, j, k);
      for (let component = 0; component < 3; component++) {
        const a = (component + 1) % 3, b = (component + 2) % 3;
        const qa = [...coordinate], qb = [...coordinate]; qa[a]!--; qb[b]!--;
        output[component]![id] = (this.edgeCoefficients[b]![id]! - this.edgeCoefficients[b]![macIndex(n, qa[0]!, qa[1]!, qa[2]!)]!
          - this.edgeCoefficients[a]![id]! + this.edgeCoefficients[a]![macIndex(n, qb[0]!, qb[1]!, qb[2]!)]!) / h + this.mean[component]!;
      }
    }
    return output;
  }
  /** Signed-time flow of one Hamiltonian pair. Positive time is forward;
   * negative time is the corresponding departure. No point is wrapped: only
   * coefficient lookup is periodic, preserving a continuous unwrapped map. */
  pair(point: Triple, component: number, dt: number, options: MapOptions = {}): MapJet {
    const maximumIterations = options.maximumIterations ?? 8, tolerance = options.residualTolerance ?? 2e-13;
    const maximumCondition = options.maximumConditionNumber ?? 100, maximumContraction = options.maximumContraction ?? .9;
    if (!point.every(Number.isFinite) || !(Number.isFinite(dt) && Number.isInteger(component) && component >= 0 && component <= 2)
      || !(Number.isInteger(maximumIterations) && maximumIterations >= 0 && maximumIterations <= 32)
      || !(Number.isFinite(tolerance) && tolerance > 0 && Number.isFinite(maximumCondition) && maximumCondition >= 1
        && maximumContraction > 0 && maximumContraction < 1)) throw new Error("invalid pair-map contract");
    // Each second potential derivative is bounded by4*C/h² from the spline
    // difference chain and partition of unity. Two entries per active row
    // give ||dt*Df/2||∞<=4|dt|C/h² globally, including support crossings.
    const contractionBound = 4 * Math.abs(dt) * this.coefficientBounds[component]! / this.grid.h ** 2;
    if (!(contractionBound < maximumContraction)) throw new Error(`pair-map global contraction bound rejected: ${contractionBound}`);
    const a = (component + 1) % 3, b = (component + 2) % 3;
    let next = [...point] as [number, number, number];
    const evaluate = (position: Triple) => {
      const da = [0, 0, 0] as [number, number, number], db = [...da] as [number, number, number]; da[a] = 1; db[b] = 1;
      const fa = this.potential(position, component, db), fb = -this.potential(position, component, da);
      const G = new Array<number>(9).fill(0);
      for (let axis = 0; axis < 3; axis++) {
        const dba = [...db] as [number, number, number], daa = [...da] as [number, number, number]; dba[axis]!++; daa[axis]!++;
        G[3 * a + axis] = this.potential(position, component, dba); G[3 * b + axis] = -this.potential(position, component, daa);
      }
      const e00 = 1 - .5 * dt * G[3 * a + a]!, e01 = -.5 * dt * G[3 * a + b]!;
      const e10 = -.5 * dt * G[3 * b + a]!, e11 = 1 - .5 * dt * G[3 * b + b]!;
      const determinant = e00 * e11 - e01 * e10;
      const inverse = [e11 / determinant, -e01 / determinant, -e10 / determinant, e00 / determinant];
      const condition = Math.max(Math.abs(e00) + Math.abs(e01), Math.abs(e10) + Math.abs(e11))
        * Math.max(Math.abs(inverse[0]!) + Math.abs(inverse[1]!), Math.abs(inverse[2]!) + Math.abs(inverse[3]!));
      if (!Number.isFinite(condition) || condition > maximumCondition) throw new Error(`pair-map conditioning rejected: ${condition}`);
      return { fa, fb, G, inverse, condition };
    };
    for (let iteration = 0; iteration <= maximumIterations; iteration++) {
      const midpoint = point.map((value, axis) => (value + next[axis]!) / 2) as unknown as Triple;
      const e = evaluate(midpoint), ra = next[a]! - point[a]! - dt * e.fa, rb = next[b]! - point[b]! - dt * e.fb;
      const residual = Math.max(Math.abs(ra), Math.abs(rb));
      if (residual <= tolerance) {
        // Implicit derivatives are returned ONLY after residual admission.
        // They approximate derivatives of the solved map, not an assertion
        // that a finite-iteration numerical map has exactly unit determinant.
        const J = [...IDENTITY3] as number[];
        for (let axis = 0; axis < 3; axis++) {
          J[3 * a + axis]! += dt * (e.inverse[0]! * e.G[3 * a + axis]! + e.inverse[1]! * e.G[3 * b + axis]!);
          J[3 * b + axis]! += dt * (e.inverse[2]! * e.G[3 * a + axis]! + e.inverse[3]! * e.G[3 * b + axis]!);
        }
        const midJ = J.map((v, i) => (v + IDENTITY3[i]!) / 2), Hfa = new Array<number>(9), Hfb = new Array<number>(9);
        for (let j = 0; j < 3; j++) for (let k = j; k < 3; k++) {
          const dfa = [0, 0, 0] as [number, number, number], dfb = [...dfa] as [number, number, number];
          dfa[b]!++; dfa[j]!++; dfa[k]!++; dfb[a]!++; dfb[j]!++; dfb[k]!++;
          Hfa[3 * j + k] = Hfa[3 * k + j] = this.potential(midpoint, component, dfa);
          Hfb[3 * j + k] = Hfb[3 * k + j] = -this.potential(midpoint, component, dfb);
        }
        const hessians = [new Array<number>(9).fill(0), new Array<number>(9).fill(0), new Array<number>(9).fill(0)];
        for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) {
          let rhsA = 0, rhsB = 0;
          for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) {
            rhsA += Hfa[3 * u + v]! * midJ[3 * u + j]! * midJ[3 * v + k]!;
            rhsB += Hfb[3 * u + v]! * midJ[3 * u + j]! * midJ[3 * v + k]!;
          }
          hessians[a]![3 * j + k] = dt * (e.inverse[0]! * rhsA + e.inverse[1]! * rhsB);
          hessians[b]![3 * j + k] = dt * (e.inverse[2]! * rhsA + e.inverse[3]! * rhsB);
        }
        return { point: next, jacobian: asMatrix(J), hessians: hessians as unknown as [Matrix3, Matrix3, Matrix3],
          residual, iterations: iteration, conditionNumber: e.condition, contractionBound,
          maximumPairPositionErrorBound: residual / (1 - contractionBound) };
      }
      if (iteration === maximumIterations) throw new Error(`pair-map unconverged after ${maximumIterations} iterations: residual ${residual}`);
      next[a]! -= e.inverse[0]! * ra + e.inverse[1]! * rb; next[b]! -= e.inverse[2]! * ra + e.inverse[3]! * rb;
    }
    throw new Error("unreachable pair-map result");
  }
  /** Palindromic second-order composition for the frozen velocity: half mean,
   * Ax/2,Ay/2,Az,Ay/2,Ax/2,half mean. It is a bounded single step, not history. */
  step(point: Triple, dt: number, options: MapOptions = {}): MapJet {
    if (!Number.isFinite(dt) || !point.every(Number.isFinite)) throw new Error("invalid map step");
    const first = point.map((v, axis) => v + .5 * dt * this.mean[axis]!) as unknown as Triple;
    let result = identityJet(first);
    for (const [component, fraction] of [[0, .5], [1, .5], [2, 1], [1, .5], [0, .5]] as const) {
      result = composeJets(this.pair(result.point, component, fraction * dt, options), result);
    }
    return { ...result, point: result.point.map((v, axis) => v + .5 * dt * this.mean[axis]!) as unknown as Triple };
  }
}
