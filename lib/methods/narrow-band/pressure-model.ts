/**
 * Float64 reference of lane P's brick-pool pressure solver (brick-abi.md
 * §6.1): MGPCG over an aggregation hierarchy keyed by cell coordinates. Every
 * GPU kernel of `pressure-*` is checked against it. It is a test oracle
 * only, never a CPU solve.
 *
 * The system is `operator`'s:
 *
 *   DIAG(c) X(c) - sum_nbr K X(nbr) = RHS(c)
 *
 * over the liquid cells, with X = p dt / rho and X = 0 at air. K_a(c) is the
 * coupling across c's lower face along a, held by the upper cell.
 *
 * The preconditioner is one V-cycle.
 * - Hierarchy: level l aggregates the cells with the same c >> l (2^3 per
 *   parent), so levels 1-3 live inside an 8^3 brick, level 3 is one row a
 *   brick, and levels 4-7 (16h ... 128h) span bricks and pages.
 * - Coarse operators: Galerkin over piecewise-constant aggregation divided by
 *   an over-correction factor, A_c = P^T A P / OC (Braess 1995), OC 2. On a
 *   7-point Laplacian, P^T A P couples two 2^3 aggregates by their 4 shared
 *   faces, where the rediscretised 2h operator under the same summed
 *   restriction couples them by 2 (McAdams, Sifakis & Teran 2010 rediscretise):
 *   OC 2 is the rediscretised operator in the interior, with Galerkin's
 *   boundary rows. OC 1.6 (the seep's) under-corrects the smoothest error by
 *   0.8 a level, compounding over seven levels: on a captured dam state it
 *   took 10-11 PCG iterations to 0.05 1/s where OC 2 takes 6-7 (lane P,
 *   2026-11-15, docs/plans/nbflip-port/progress-lane-p1.md item 4a).
 * - Coarsest level: solved exactly by Gauss-Jordan (Trottenberg, Oosterlee &
 *   Schuller 2001, sec. 2.5; Golub & Van Loan 2013, sec. 4.2). It is the
 *   first level with at most `coarsestRows` rows, capped at MAX_COARSEST_ROWS.
 * - Level 0 smoother: a staggered block Schwarz (Smith, Bjorstad & Gropp
 *   1996, ch. 1-2). It is block red-black Gauss-Seidel over the 8^3 bricks,
 *   then over the blocks shifted by (4, 0, 4), each with its halo frozen at
 *   the pass's entry. Post-smoothing applies the adjoint: the passes in
 *   reverse order, each block's colours reversed.
 * - Levels 1 and up: block red-black Gauss-Seidel over the level's cells
 *   inside each 8h block (a brick at levels 1-2, a 2^3 group above), with
 *   the same pre/post adjoint rule.
 * - So the V-cycle is a symmetric positive definite operator, and CG
 *   applies (tests/pressure-model.test.ts checks it).
 *
 * PCG (Saad 2003, Alg. 9.1), with the convergence check before the V-cycle:
 * - An iteration whose max-norm residual is at the tolerance stops before
 *   any preconditioner work.
 * - The residual is recomputed as b - A x every `replaceEvery` iterations
 *   (residual replacement, van der Vorst & Ye 2000). The recurrence
 *   r - alpha q drifts, as the arena's GCR(1) did: lane P1's finding, 2026-11-15.
 */

export const OC = 2;
export const MAX_COARSEST_ROWS = 64;
const AX: readonly (readonly [number, number, number])[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
/** A cell coordinate packed into one number (13 bits an axis, offset so -1 packs). */
const B = 16384, OFF = 8;
export const packCell = (x: number, y: number, z: number) => (x + OFF) + B * ((y + OFF) + B * (z + OFF));
export const unpackCell = (k: number): [number, number, number] => [k % B - OFF, Math.floor(k / B) % B - OFF, Math.floor(k / (B * B)) - OFF];

export interface ModelLevel {
  readonly shift: number;
  /** Packed coordinates of the level's rows, sorted. */
  readonly keys: readonly number[];
  readonly index: ReadonlyMap<number, number>;
  readonly diag: Float64Array;
  /** k[a][i]: the coupling across row i's lower face along a (to the row at c - e_a), 0 if none. */
  readonly k: readonly [Float64Array, Float64Array, Float64Array];
}

export interface ModelOptions {
  /** Rows at most of the coarsest level (≤ MAX_COARSEST_ROWS). */
  readonly coarsestRows?: number;
  /** Red-black sweeps inside a level 0 Schwarz block. */
  readonly fineSweeps?: number;
  /** Red-black sweeps inside a coarse block. */
  readonly coarseSweeps?: number;
  /** false: level 0 takes the bricks alone (one covering). */
  readonly staggered?: boolean;
  /** A fixed level count (L0 .. L(levels-1)), as the GPU solver builds it; its coarsest must hold at most
   * `coarsestRows` rows. 0 (the default): the first level with at most `coarsestRows` rows is the coarsest. */
  readonly levels?: number;
}

/** Level 0 from per-cell rows: diag, the three lower couplings and RHS, keyed by packed cell. */
export interface FineRows { readonly keys: readonly number[]; readonly diag: ArrayLike<number>; readonly k: readonly [ArrayLike<number>, ArrayLike<number>, ArrayLike<number>] }

function makeLevel(shift: number, keys: number[]): ModelLevel {
  keys.sort((a, b) => a - b);
  const index = new Map<number, number>(); keys.forEach((k, i) => index.set(k, i));
  const n = keys.length;
  return { shift, keys, index, diag: new Float64Array(n), k: [new Float64Array(n), new Float64Array(n), new Float64Array(n)] };
}

/** Galerkin aggregation of F by 2^3 over OC. */
export function coarsenLevel(F: ModelLevel, oc = OC): ModelLevel {
  const parentKey = (k: number) => { const [x, y, z] = unpackCell(k); return packCell(x >> 1, y >> 1, z >> 1); };
  const C = makeLevel(F.shift + 1, [...new Set(F.keys.map(parentKey))]);
  F.keys.forEach((key, i) => {
    const p = C.index.get(parentKey(key))!;
    (C.diag as Float64Array)[p]! += F.diag[i]!;
    const [x, y, z] = unpackCell(key);
    for (let a = 0; a < 3; a++) {
      const w = F.k[a]![i]!; if (!w) continue;
      const d = AX[a]!, lower = F.index.get(packCell(x - d[0], y - d[1], z - d[2]));
      if (lower === undefined) continue;
      const q = C.index.get(parentKey(F.keys[lower]!))!;
      if (q === p) (C.diag as Float64Array)[p]! -= 2 * w;
      else (C.k[a] as Float64Array)[p]! += w; // the lower neighbour's parent is p - e_a
    }
  });
  for (let i = 0; i < C.keys.length; i++) { (C.diag as Float64Array)[i]! /= oc; for (let a = 0; a < 3; a++) (C.k[a] as Float64Array)[i]! /= oc; }
  return C;
}

/** The six neighbours of row i: [row index or -1, coupling]. */
function neighbours(L: ModelLevel, i: number): [number, number][] {
  const [x, y, z] = unpackCell(L.keys[i]!), out: [number, number][] = [];
  for (let a = 0; a < 3; a++) {
    const d = AX[a]!;
    const lo = L.index.get(packCell(x - d[0], y - d[1], z - d[2]));
    if (lo !== undefined && L.k[a]![i]) out.push([lo, L.k[a]![i]!]);
    const hi = L.index.get(packCell(x + d[0], y + d[1], z + d[2]));
    if (hi !== undefined && L.k[a]![hi]) out.push([hi, L.k[a]![hi]!]);
  }
  return out;
}

export function applyLevel(L: ModelLevel, x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    let s = L.diag[i]! * x[i]!;
    for (const [j, w] of neighbours(L, i)) s -= w * x[j]!;
    out[i] = s;
  }
  return out;
}

/** Block red-black Gauss-Seidel: blocks of side `side` cells at offset `off`, the halo frozen on entry. */
function blockGS(L: ModelLevel, e: Float64Array, b: Float64Array, side: number, off: readonly number[], colours: readonly number[]) {
  const frozen = Float64Array.from(e);
  const block = (k: number) => { const c = unpackCell(k); return c.map((v, a) => Math.floor((v + off[a]!) / side)).join(","); };
  const blocks = L.keys.map(block), parity = L.keys.map(k => { const [x, y, z] = unpackCell(k); return (x + y + z) & 1; });
  for (const col of colours) {
    for (let i = 0; i < e.length; i++) {
      if (parity[i] !== col || !(L.diag[i]! > 0)) continue;
      let s = b[i]!;
      for (const [j, w] of neighbours(L, i)) s += w * (blocks[j] === blocks[i] ? e[j]! : frozen[j]!);
      e[i] = s / L.diag[i]!;
    }
  }
}

/** The dense inverse of a level's operator (Gauss-Jordan without pivoting; null on a pivot ≤ 0: FAIL.PIVOT). */
export function denseInverse(L: ModelLevel): Float64Array | null {
  const m = L.keys.length, A = new Float64Array(m * m);
  for (let i = 0; i < m; i++) { A[i * m + i] = L.diag[i]!; for (const [j, w] of neighbours(L, i)) A[i * m + j]! -= w; }
  for (let p = 0; p < m; p++) {
    const app = A[p * m + p]!; if (!(app > 0)) return null;
    const r = 1 / app, rowP = A.slice(p * m, p * m + m), colP = new Float64Array(m);
    for (let i = 0; i < m; i++) colP[i] = A[i * m + p]!;
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) {
      const t = i * m + j;
      if (i === p && j === p) A[t] = r; else if (i === p) A[t] = rowP[j]! * r; else if (j === p) A[t] = -colP[i]! * r; else A[t] = A[t]! - colP[i]! * rowP[j]! * r;
    }
  }
  return A;
}

const rb = (n: number, rev: boolean) => Array.from({ length: 2 * n }, (_, s) => rev ? 1 - (s & 1) : s & 1);
const ZERO = [0, 0, 0], SHIFT = [4, 0, 4];

export class PressureModel {
  readonly levels: ModelLevel[];
  readonly inverse: Float64Array;
  readonly options: Required<ModelOptions>;
  constructor(fine: FineRows, options: ModelOptions = {}) {
    this.options = { coarsestRows: MAX_COARSEST_ROWS, fineSweeps: 3, coarseSweeps: 2, staggered: true, levels: 0, ...options };
    if (this.options.coarsestRows > MAX_COARSEST_ROWS) throw new Error(`Pressure model: a coarsest level of ${this.options.coarsestRows} rows, over ${MAX_COARSEST_ROWS}`);
    const L0 = makeLevel(0, [...fine.keys]);
    fine.keys.forEach((key, s) => { const i = L0.index.get(key)!; (L0.diag as Float64Array)[i] = fine.diag[s]!; for (let a = 0; a < 3; a++) (L0.k[a] as Float64Array)[i] = fine.k[a]![s]!; });
    this.levels = [L0];
    const fixed = this.options.levels;
    if (fixed) {
      if (!Number.isInteger(fixed) || fixed < 2) throw new Error(`Pressure model: ${fixed} levels`);
      while (this.levels.length < fixed) this.levels.push(coarsenLevel(this.levels.at(-1)!));
      const top = this.levels.at(-1)!.keys.length;
      if (top > this.options.coarsestRows) throw new Error(`Pressure model: a coarsest level of ${top} rows at ${fixed} levels, over ${this.options.coarsestRows}`);
    } else while (this.levels.at(-1)!.keys.length > this.options.coarsestRows || this.levels.length < 2) this.levels.push(coarsenLevel(this.levels.at(-1)!));
    const inv = denseInverse(this.levels.at(-1)!);
    if (!inv) throw new Error("Pressure model: the coarsest operator has a non-positive pivot");
    this.inverse = inv;
  }
  get fine(): ModelLevel { return this.levels[0]!; }
  /** One V-cycle from zero: z ≈ A^-1 r. */
  vcycle(r: Float64Array, l = 0, trace?: (level: number, stage: "b" | "pre" | "post", v: Float64Array) => void): Float64Array {
    const L = this.levels[l]!, top = this.levels.length - 1;
    if (l === top) { const m = L.keys.length, z = new Float64Array(m); for (let i = 0; i < m; i++) { let s = 0; for (let j = 0; j < m; j++) s += this.inverse[i * m + j]! * r[j]!; z[i] = s; } return z; }
    trace?.(l, "b", Float64Array.from(r));
    const e = new Float64Array(r.length), o = this.options;
    const smooth = (post: boolean) => {
      if (l === 0) {
        const passes = o.staggered ? [ZERO, SHIFT, ZERO] : [ZERO];
        for (const off of passes) blockGS(L, e, r, 8, off, rb(o.fineSweeps, post));
      } else blockGS(L, e, r, Math.max(2, 8 >> l), ZERO, rb(o.coarseSweeps, post));
    };
    smooth(false);
    trace?.(l, "pre", Float64Array.from(e));
    const Ae = applyLevel(L, e), res = r.map((v, i) => v - Ae[i]!);
    const C = this.levels[l + 1]!, rc = new Float64Array(C.keys.length);
    const parent = L.keys.map(k => { const [x, y, z] = unpackCell(k); return C.index.get(packCell(x >> 1, y >> 1, z >> 1))!; });
    for (let i = 0; i < res.length; i++) rc[parent[i]!]! += res[i]!;
    const ec = this.vcycle(rc, l + 1, trace);
    for (let i = 0; i < e.length; i++) e[i]! += ec[parent[i]!]!;
    smooth(true);
    trace?.(l, "post", Float64Array.from(e));
    return e;
  }
  /** MGPCG (check before the V-cycle). Returns the iterate and the iterations whose V-cycle ran. */
  solve(b: Float64Array, x0: Float64Array, tol: number, maxIterations: number, replaceEvery = 1): { x: Float64Array; iterations: number; residual: number } {
    const A = (v: Float64Array) => applyLevel(this.fine, v), x = Float64Array.from(x0);
    const linf = (v: Float64Array) => v.reduce((m, t) => Math.max(m, Math.abs(t)), 0);
    const dot = (u: Float64Array, v: Float64Array) => { let s = 0; for (let i = 0; i < u.length; i++) s += u[i]! * v[i]!; return s; };
    let r = (() => { const Ax = A(x); return b.map((v, i) => v - Ax[i]!); })();
    let p = new Float64Array(b.length), rz = 0;
    for (let k = 0; k < maxIterations; k++) {
      if (linf(r) <= tol) return { x, iterations: k, residual: linf(r) };
      const z = this.vcycle(r), rz1 = dot(r, z), beta = k === 0 ? 0 : rz1 / rz; rz = rz1;
      p = p.map((v, i) => z[i]! + beta * v);
      const q = A(p), alpha = rz / dot(p, q);
      for (let i = 0; i < x.length; i++) x[i]! += alpha * p[i]!;
      if ((k + 1) % replaceEvery === 0) { const Ax = A(x); r = b.map((v, i) => v - Ax[i]!); }
      else for (let i = 0; i < r.length; i++) r[i]! -= alpha * q[i]!;
    }
    return { x, iterations: maxIterations, residual: linf(r) };
  }
}
