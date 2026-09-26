import { uniformMixedPressureLevel, type UniformMixedLayout } from "../../lib/methods/uniform/uniform-mixed-layout";
import { DEFAULT_UNIFORM_CM11A_SCHEDULE, type UniformCM11aSchedule } from "../../lib/methods/uniform/pressure-policy";
import { mixedPressureFixture, factorPressureCore, type PressureFixture } from "./uniform-mixed-pressure";

type Rows = Map<number, number>[];
function add(row: Map<number, number>, column: number, value: number) { row.set(column, (row.get(column) ?? 0) + value); }
function apply(rows: Rows, values: ArrayLike<number>): Float64Array {
  return Float64Array.from(rows, row => [...row].reduce((sum, [j, w]) => sum + w * values[j]!, 0));
}
function operators(fixture: PressureFixture): { core: Rows; full: Rows } {
  const core: Rows = fixture.cells.map(() => new Map()), full: Rows = fixture.cells.map(() => new Map());
  fixture.faces.forEach((face, k) => {
    for (const [i, j, sign] of [[face.left, face.right, -1], [face.right, face.left, 1]]) {
      add(core[i!]!, i!, face.area / face.distance); add(core[i!]!, j!, -face.area / face.distance);
      for (const [donor, weight] of fixture.gradients[k]!) add(full[i!]!, donor, sign! * face.area * weight);
    }
  });
  return { core, full };
}

/** Closed, fully liquid CPU oracle of the existing CM11a cycle order on the
 * mixed hierarchy. Explicit matrices and an exact bottom solve are test-only;
 * they are not a runtime solver, a free-surface test, or a GPU timing claim.
 * Vectors are extensive (cell-volume-scaled), so restriction is a sum.
 */
export class MixedMultigridOracle {
  readonly layouts: readonly UniformMixedLayout[];
  readonly fixtures: readonly PressureFixture[];
  readonly parents: readonly Uint32Array[];
  readonly interpolation: readonly Rows[];
  private readonly rows: readonly { core: Rows; full: Rows }[];
  private readonly colours: readonly Uint8Array[];
  private readonly solveBottom: (rhs: ArrayLike<number>) => Float64Array;
  /** "native-jacobi" is the production smoother: two simultaneous projected
   * updates per sweep at the native 2/3 weight, reconstruction frozen once. */
  constructor(layout: UniformMixedLayout, private readonly schedule: UniformCM11aSchedule = DEFAULT_UNIFORM_CM11A_SCHEDULE,
    private readonly smoother: "gauss-seidel" | "native-jacobi" = "gauss-seidel") {
    this.layouts = [layout, uniformMixedPressureLevel(layout, 2), uniformMixedPressureLevel(layout, 4)];
    this.fixtures = this.layouts.map(mixedPressureFixture); this.rows = this.fixtures.map(operators);
    const h = layout.lattice.cellSize_m;
    this.colours = this.fixtures.map(f => Uint8Array.from(f.cells, cell => 2 * Math.log2(cell.width)
      + (cell.center.reduce((sum, v, a) => sum + Math.floor(v / h[a]! / cell.width), 0) & 1)));
    const owner = (level: number, point: readonly number[]) => {
      const i = this.fixtures[level]!.cells.findIndex(c => c.center.every((v, a) => Math.abs(v - point[a]!) < .5 * c.width * h[a]!));
      if (i < 0) throw new Error("Missing pressure transfer owner");
      return i;
    };
    this.parents = this.fixtures.slice(0, -1).map((f, level) => Uint32Array.from(f.cells, c => owner(level + 1, c.center)));
    this.interpolation = this.fixtures.slice(0, -1).map((f, level) => f.cells.map((cell, i) => {
      const parent = this.parents[level]![i]!, width = this.fixtures[level + 1]!.cells[parent]!.width;
      if (width === cell.width) return new Map([[parent, 1]]);
      const q = cell.center.map((v, a) => v / (h[a]! * width) - .5), base = q.map(Math.floor);
      const fraction = q.map((v, a) => v - base[a]!);
      const row = new Map<number, number>(); let total = 0;
      for (let k = 0; k < 8; k++) {
        const bits = [k & 1, (k >> 1) & 1, k >> 2];
        const point = bits.map((bit, a) => (base[a]! + bit + .5) * width * h[a]!);
        // Match native trilinear prolongation: omit out-of-domain taps and
        // renormalize, rather than sampling a persistent zero-pressure halo.
        if (point.some((v, a) => v < 0 || v >= layout.lattice.dimensions[a]! * h[a]!)) continue;
        const weight = bits.reduce((w, bit, a) => w * (bit ? fraction[a]! : 1 - fraction[a]!), 1);
        add(row, owner(level + 1, point), weight); total += weight;
      }
      if (!total) throw new Error("Empty pressure interpolation");
      for (const [j, w] of row) row.set(j, w / total);
      return row;
    }));
    this.solveBottom = factorPressureCore(this.fixtures[2]!);
  }
  apply(level: number, pressure: ArrayLike<number>): Float64Array { return apply(this.rows[level]!.full, pressure); }
  restrict(level: number, residual: ArrayLike<number>): Float64Array {
    const result = new Float64Array(this.fixtures[level + 1]!.cells.length);
    this.parents[level]!.forEach((parent, i) => { result[parent]! += residual[i]!; });
    return result;
  }
  prolong(level: number, pressure: ArrayLike<number>): Float64Array { return apply(this.interpolation[level]!, pressure); }
  smoothSweep(level: number, pressure: ArrayLike<number>, rhs: ArrayLike<number>): Float64Array {
    const result = Float64Array.from(pressure); this.smooth(level, result, rhs, 1); return result;
  }
  private smooth(level: number, pressure: Float64Array, rhs: ArrayLike<number>, sweeps: number) {
    const { core, full } = this.rows[level]!;
    for (let sweep = 0; sweep < sweeps; sweep++) {
      const coreP = apply(core, pressure), fullP = apply(full, pressure);
      // Freeze only reconstruction. Six tier/parity colours separate every
      // direct core edge; wider reconstruction edges are never read in-place.
      const correctedRhs = coreP.map((v, i) => rhs[i]! + v - fullP[i]!);
      if (this.smoother === "native-jacobi") {
        for (let half = 0; half < 2; half++) {
          const old = Float64Array.from(pressure);
          for (let i = 0; i < pressure.length; i++) {
            let sum = 0;
            for (const [j, w] of core[i]!) if (i !== j) sum += w * old[j]!;
            const diagonal = core[i]!.get(i) ?? 0;
            pressure[i] = diagonal > 0 ? old[i]! + (2 / 3) * ((correctedRhs[i]! - sum) / diagonal - old[i]!) : 0;
          }
        }
        continue;
      }
      for (let colour = 0; colour < 6; colour++) for (let i = 0; i < pressure.length; i++) {
        if (this.colours[level]![i] !== colour) continue;
        let sum = 0;
        for (const [j, w] of core[i]!) if (i !== j) sum += w * pressure[j]!;
        const diagonal = core[i]!.get(i) ?? 0;
        pressure[i] = diagonal > 0 ? (correctedRhs[i]! - sum) / diagonal : 0;
      }
    }
  }
  private residual(level: number, pressure: ArrayLike<number>, rhs: ArrayLike<number>): Float64Array {
    return this.apply(level, pressure).map((v, i) => rhs[i]! - v);
  }
  private vCycle(level: number, pressure: Float64Array, rhs: ArrayLike<number>): Float64Array {
    if (level === 2) return this.solveBottom(rhs);
    this.smooth(level, pressure, rhs, this.schedule.preSweeps);
    const coarseRhs = this.restrict(level, this.residual(level, pressure, rhs));
    const correction = this.vCycle(level + 1, new Float64Array(coarseRhs.length), coarseRhs);
    const fineCorrection = this.prolong(level, correction);
    pressure.forEach((v, i) => { pressure[i] = v + fineCorrection[i]!; });
    this.smooth(level, pressure, rhs, this.schedule.postSweeps);
    return pressure;
  }
  private fullCycle(pressure: Float64Array, rhs: ArrayLike<number>): Float64Array {
    const pyramid = [this.residual(0, pressure, rhs)];
    for (let level = 0; level < 2; level++) pyramid.push(this.restrict(level, pyramid[level]!));
    let correction = this.solveBottom(pyramid[2]!);
    for (let level = 1; level >= 0; level--) correction = this.vCycle(level, this.prolong(level, correction), pyramid[level]!);
    return pressure.map((v, i) => v + correction[i]!);
  }
  solve(rhs: ArrayLike<number>): { pressure: Float64Array; residuals: readonly number[] } {
    let pressure: Float64Array = new Float64Array(this.fixtures[0]!.cells.length);
    const residuals: number[] = [];
    for (let cycle = 0; cycle < this.schedule.fullCycles + this.schedule.vCycles; cycle++) {
      pressure = cycle < this.schedule.fullCycles ? this.fullCycle(pressure, rhs) : this.vCycle(0, pressure, rhs);
      residuals.push(Math.max(...this.residual(0, pressure, rhs).map((v, i) => Math.abs(v) / this.fixtures[0]!.cells[i]!.volume)));
    }
    return { pressure, residuals };
  }
}
