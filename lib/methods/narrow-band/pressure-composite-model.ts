/**
 * Float64 reference of lane P's composite MGPCG: the composite rows of pressure-composite.wgsl.ts (the
 * h rows of h-owned tiles and the native 4h rows) solved by PCG (Saad 2003, Alg. 9.1) under one
 * symmetric V-cycle (McAdams, Sifakis & Teran 2010). It is a test oracle only, never a CPU solve.
 *
 * The system: an h row c holds DIAG, K[a] (its lower face along a) and KU[a] (its upper face, only
 * toward a native row). A K[a] couples to the h row c - e_a, or to the native row of the tile holding
 * c - e_a; a KU[a] couples to the native row of the tile holding c + e_a. A native row T holds DIAG
 * and K[a], its lower face to the native row T - e_a. Its seam couplings are its h neighbours' K and
 * KU, so the matrix is symmetric by construction.
 *
 * The hierarchy aggregates by coordinate, as pressure-model.ts does (Braess 1995, OC 2):
 * - L0 holds the h rows (smoothed) and the native rows (carried by the h passes, smoothed by their own pass).
 * - L1 holds the 2h aggregates of the h rows (smoothed), and the native rows carried again, as at L0.
 * - L2 is the 4h tiles: the h tiles' aggregates and the native rows, smoothed together.
 * - L3 and up aggregate every L2 row by 2^3. The coarsest level is inverted exactly.
 * - Coarse operators are A_c = W^-1 P^T A P W^-1, where P sums by aggregation (identity on a carried
 *   native row) and W is sqrt(OC) on an aggregated row and 1 on a carried one. So the correction is
 *   P W (P^T A P)^-1 W P^T. Aggregates are over-corrected by OC, as the pool MGPCG's /OC, and native rows
 *   are not. The congruence keeps A_c symmetric positive definite.
 * - Smoothers: level 0 runs the staggered block Schwarz over the h rows (bricks, then the blocks
 *   shifted by (4, 0, 4), then the bricks). Levels 1 and up run block red-black Gauss-Seidel in blocks of
 *   side max(2, 8 >> l). Post-smoothing is the adjoint, and carried rows are held at their current
 *   value. At L0 and L1 the carried native rows then get their own pass (nativeGS: block red-black GS in
 *   2^3-tile blocks, the h rows and other blocks frozen at the pass's entry), after the h passes in
 *   pre-smoothing and before them, colours reversed, in post-smoothing. This is the composite-grid
 *   smoother of the fast adaptive composite method (FAC, McCormick & Thomas 1986):
 *   every composite row is relaxed on its own level before the residual is restricted. Carried unsmoothed,
 *   a native row's error reached only L2's smoother, a level above its own, and a captured tiered dam
 *   state took 9-12 PCG iterations to 0.05 1/s where the pass takes 7-8 (lane P item 4a). The pass and its
 *   adjoint keep the V-cycle symmetric positive definite.
 * - The convergence check runs before each V-cycle: max|r| <= tol, with a native row's residual taken
 *   over 64, its band-equivalent (a native row is 64 h rows' worth: RHS -64 h^2 div).
 */
import { MAX_COARSEST_ROWS, OC, packCell, unpackCell } from "./pressure-model";

type V3 = [number, number, number];
export interface CompositeRows {
  /** h rows: cell coordinates, DIAG, K[a] (lower), KU[a] (upper, native only). */
  readonly f: { readonly cells: readonly V3[]; readonly diag: ArrayLike<number>; readonly k: readonly ArrayLike<number>[]; readonly ku: readonly ArrayLike<number>[] };
  /** Native rows: tile coordinates, DIAG, K[a] (lower, to a native row). */
  readonly n: { readonly tiles: readonly V3[]; readonly diag: ArrayLike<number>; readonly k: readonly ArrayLike<number>[] };
}
interface Level {
  readonly scale: number;
  readonly keys: number[];
  readonly coords: V3[];
  /** Whether a row is a carried native row (L0 and L1), which this level does not smooth. */
  readonly carried: boolean[];
  readonly index: Map<number, number>;
  readonly diag: Float64Array;
  readonly nbrs: [number, number][][];
}
const key = (c: V3, native: boolean) => 2 * packCell(c[0], c[1], c[2]) + (native ? 1 : 0);
const shr = (c: V3, n: number): V3 => [c[0] >> n, c[1] >> n, c[2] >> n];
const unit = (a: number): V3 => [a === 0 ? 1 : 0, a === 1 ? 1 : 0, a === 2 ? 1 : 0];
const add = (u: V3, v: V3, s = 1): V3 => [u[0] + s * v[0], u[1] + s * v[1], u[2] + s * v[2]];

function level(scale: number, rows: { key: number; coord: V3; carried: boolean }[]): Level {
  rows.sort((a, b) => a.key - b.key);
  const index = new Map<number, number>(); rows.forEach((r, i) => index.set(r.key, i));
  return { scale, keys: rows.map(r => r.key), coords: rows.map(r => r.coord), carried: rows.map(r => r.carried), index, diag: new Float64Array(rows.length), nbrs: rows.map(() => []) };
}

/** The composite L0 from the rows. Throws on an asymmetric or dangling coupling. */
export function compositeLevel0(rows: CompositeRows): Level {
  const F = rows.f, Nr = rows.n;
  const native = new Set(Nr.tiles.map(t => packCell(t[0], t[1], t[2])));
  const isNative = (c: V3) => native.has(packCell(...shr(c, 2)));
  const L = level(0, [...F.cells.map(c => ({ key: key(c, false), coord: c, carried: false })), ...Nr.tiles.map(t => ({ key: key(t, true), coord: t, carried: true }))]);
  const link = new Map<string, number>();
  const put = (i: number, j: number, w: number) => {
    if (!w) return;
    const id = i < j ? `${i},${j}` : `${j},${i}`, old = link.get(id);
    if (old === undefined) link.set(id, w); else if (Math.abs(old - w) > 1e-6 * Math.max(Math.abs(old), Math.abs(w))) throw new Error(`Composite model: an asymmetric coupling ${id}: ${old} against ${w}`);
  };
  F.cells.forEach((c, s) => {
    const i = L.index.get(key(c, false))!; L.diag[i] = F.diag[s]!;
    for (let a = 0; a < 3; a++) {
      const lo = add(c, unit(a), -1), up = add(c, unit(a));
      const wl = F.k[a]![s]!;
      if (wl) { const j = isNative(lo) ? L.index.get(key(shr(lo, 2), true)) : L.index.get(key(lo, false)); if (j === undefined) throw new Error(`Composite model: h row ${c} couples below to no row`); put(i, j, wl); }
      const wu = F.ku[a]![s]!;
      if (wu) { const j = L.index.get(key(shr(up, 2), true)); if (j === undefined || !isNative(up)) throw new Error(`Composite model: h row ${c} couples above to no native row`); put(i, j, wu); }
    }
  });
  Nr.tiles.forEach((t, s) => {
    const i = L.index.get(key(t, true))!; L.diag[i] = Nr.diag[s]!;
    for (let a = 0; a < 3; a++) { const w = Nr.k[a]![s]!; if (!w) continue; const j = L.index.get(key(add(t, unit(a), -1), true)); if (j === undefined) throw new Error(`Composite model: native row ${t} couples below to no native row`); put(i, j, w); }
  });
  for (const [id, w] of link) { const [i, j] = id.split(",").map(Number) as [number, number]; L.nbrs[i]!.push([j, w]); L.nbrs[j]!.push([i, w]); }
  return L;
}

/** The next level: L0 -> L1 and L1 -> L2 aggregate h rows and carry native rows; L2 up aggregates all. */
function coarsen(F: Level): { C: Level; parent: number[] } {
  const into = (i: number): { key: number; coord: V3; carried: boolean; w: number } => {
    const c = F.coords[i]!, nat = (F.keys[i]! & 1) === 1;
    if (F.scale < 2 && nat) return { key: F.keys[i]!, coord: c, carried: F.scale + 1 < 2, w: 1 };
    const p = shr(c, 1); return { key: key(p, false), coord: p, carried: false, w: Math.sqrt(OC) };
  };
  const targets = F.keys.map((_, i) => into(i));
  const seen = new Map<number, { key: number; coord: V3; carried: boolean }>();
  for (const t of targets) if (!seen.has(t.key)) seen.set(t.key, t);
  const C = level(F.scale + 1, [...seen.values()]);
  const parent = targets.map(t => C.index.get(t.key)!), w = new Float64Array(C.keys.length);
  targets.forEach((t, i) => { w[parent[i]!] = t.w; });
  const acc = new Map<string, number>();
  for (let i = 0; i < F.keys.length; i++) {
    const p = parent[i]!; C.diag[p]! += F.diag[i]!;
    for (const [j, v] of F.nbrs[i]!) { const q = parent[j]!; if (q === p) C.diag[p]! -= v; else if (p < q) acc.set(`${p},${q}`, (acc.get(`${p},${q}`) ?? 0) + v); }
  }
  for (let p = 0; p < C.keys.length; p++) C.diag[p]! /= w[p]! * w[p]!;
  for (const [id, v] of acc) { const [p, q] = id.split(",").map(Number) as [number, number]; const s = v / (w[p]! * w[q]!); C.nbrs[p]!.push([q, s]); C.nbrs[q]!.push([p, s]); }
  return { C, parent };
}

function apply(L: Level, x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) { let s = L.diag[i]! * x[i]!; for (const [j, w] of L.nbrs[i]!) s -= w * x[j]!; out[i] = s; }
  return out;
}

function blockGS(L: Level, e: Float64Array, b: Float64Array, side: number, off: readonly number[], colours: readonly number[]) {
  const frozen = Float64Array.from(e);
  const block = L.coords.map(c => c.map((v, a) => Math.floor((v + off[a]!) / side)).join(","));
  const parity = L.coords.map(c => (c[0] + c[1] + c[2]) & 1);
  for (const col of colours) for (let i = 0; i < e.length; i++) {
    if (L.carried[i] || parity[i] !== col || !(L.diag[i]! > 0)) continue;
    let s = b[i]!;
    for (const [j, w] of L.nbrs[i]!) s += w * (!L.carried[j] && block[j] === block[i] ? e[j]! : frozen[j]!);
    e[i] = s / L.diag[i]!;
  }
}

function denseInverse(L: Level): Float64Array {
  const m = L.keys.length, A = new Float64Array(m * m);
  for (let i = 0; i < m; i++) { A[i * m + i] = L.diag[i]!; for (const [j, w] of L.nbrs[i]!) A[i * m + j]! -= w; }
  for (let p = 0; p < m; p++) {
    const app = A[p * m + p]!; if (!(app > 0)) throw new Error("Composite model: a non-positive coarsest pivot");
    const r = 1 / app, rowP = A.slice(p * m, p * m + m), colP = new Float64Array(m);
    for (let i = 0; i < m; i++) colP[i] = A[i * m + p]!;
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) { const t = i * m + j; A[t] = i === p && j === p ? r : i === p ? rowP[j]! * r : j === p ? -colP[i]! * r : A[t]! - colP[i]! * rowP[j]! * r; }
  }
  return A;
}

/** The native rows' own pass at L0 and L1 (carried by the h passes): block red-black GS in 2^3-tile blocks, the
 * h rows and the other blocks' native rows frozen at the pass's entry. */
function nativeGS(L: Level, e: Float64Array, b: Float64Array, colours: readonly number[]) {
  const frozen = Float64Array.from(e), nat = (i: number) => (L.keys[i]! & 1) === 1;
  const block = L.coords.map((c, i) => nat(i) ? c.map(v => v >> 1).join(",") : "h");
  const parity = L.coords.map(c => (c[0] + c[1] + c[2]) & 1);
  for (const col of colours) for (let i = 0; i < e.length; i++) {
    if (!nat(i) || parity[i] !== col || !(L.diag[i]! > 0)) continue;
    let s = b[i]!;
    for (const [j, w] of L.nbrs[i]!) s += w * (nat(j) && block[j] === block[i] ? e[j]! : frozen[j]!);
    e[i] = s / L.diag[i]!;
  }
}

const rb = (n: number, rev: boolean) => Array.from({ length: 2 * n }, (_, s) => rev ? 1 - (s & 1) : s & 1);

export class CompositePressureModel {
  readonly levels: Level[] = [];
  readonly parents: number[][] = [];
  readonly inverse: Float64Array;
  constructor(rows: CompositeRows, readonly options: { levels: number; fineSweeps: number; coarseSweeps: number; nativeSweeps?: number } = { levels: 8, fineSweeps: 3, coarseSweeps: 2, nativeSweeps: 2 }) {
    this.levels.push(compositeLevel0(rows));
    while (this.levels.length < options.levels) { const { C, parent } = coarsen(this.levels.at(-1)!); this.levels.push(C); this.parents.push(parent); }
    if (this.levels.at(-1)!.keys.length > MAX_COARSEST_ROWS) throw new Error(`Composite model: ${this.levels.at(-1)!.keys.length} coarsest rows`);
    this.inverse = denseInverse(this.levels.at(-1)!);
  }
  get fine(): Level { return this.levels[0]!; }
  /** The row index of an h cell or a native tile at L0. */
  rowOf(c: V3, native: boolean): number | undefined { return this.fine.index.get(key(c, native)); }
  isNative(i: number): boolean { return (this.fine.keys[i]! & 1) === 1; }
  vcycle(r: Float64Array, l = 0): Float64Array {
    const L = this.levels[l]!;
    if (l === this.levels.length - 1) { const m = L.keys.length, z = new Float64Array(m); for (let i = 0; i < m; i++) { let s = 0; for (let j = 0; j < m; j++) s += this.inverse[i * m + j]! * r[j]!; z[i] = s; } return z; }
    const e = new Float64Array(r.length), o = this.options;
    const smooth = (post: boolean) => {
      const native = l < 2 ? rb(o.nativeSweeps ?? 2, post) : undefined;
      if (native && post) nativeGS(L, e, r, native);
      if (l === 0) for (const off of [[0, 0, 0], [4, 0, 4], [0, 0, 0]]) blockGS(L, e, r, 8, off, rb(o.fineSweeps, post));
      else blockGS(L, e, r, Math.max(2, 8 >> l), [0, 0, 0], rb(o.coarseSweeps, post));
      if (native && !post) nativeGS(L, e, r, native);
    };
    smooth(false);
    const Ae = apply(L, e), parent = this.parents[l]!, rc = new Float64Array(this.levels[l + 1]!.keys.length);
    for (let i = 0; i < r.length; i++) rc[parent[i]!]! += r[i]! - Ae[i]!;
    const ec = this.vcycle(rc, l + 1);
    for (let i = 0; i < e.length; i++) e[i]! += ec[parent[i]!]!;
    smooth(true);
    return e;
  }
  /** The band-equivalent max-norm of a residual (native rows over 64). */
  norm(r: Float64Array): number { let m = 0; for (let i = 0; i < r.length; i++) m = Math.max(m, Math.abs(r[i]!) / (this.isNative(i) ? 64 : 1)); return m; }
  apply(x: Float64Array): Float64Array { return apply(this.fine, x); }
  solve(b: Float64Array, x0: Float64Array, tol: number, maxIterations: number): { x: Float64Array; iterations: number; residual: number } {
    const x = Float64Array.from(x0), dot = (u: Float64Array, v: Float64Array) => { let s = 0; for (let i = 0; i < u.length; i++) s += u[i]! * v[i]!; return s; };
    const resid = () => { const Ax = this.apply(x); return b.map((v, i) => v - Ax[i]!); };
    let r = resid(), p = new Float64Array(b.length), rz = 0;
    for (let k = 0; k < maxIterations; k++) {
      if (this.norm(r) <= tol) return { x, iterations: k, residual: this.norm(r) };
      const z = this.vcycle(r), rz1 = dot(r, z), beta = k === 0 ? 0 : rz1 / rz; rz = rz1;
      p = p.map((v, i) => z[i]! + beta * v);
      const q = this.apply(p), alpha = rz / dot(p, q);
      for (let i = 0; i < x.length; i++) x[i]! += alpha * p[i]!;
      r = resid();
    }
    return { x, iterations: maxIterations, residual: this.norm(r) };
  }
}

export { unpackCell };
