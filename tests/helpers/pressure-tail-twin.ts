/** A CPU twin of the pressure arena's coarse tail (uniform-pressure-arena.ts):
 * the 32h rows, one a page, and the correction they take in each 8h cycle.
 * "pages" is today's tail: red-black sweeps from zero over every 32h row in
 * one workgroup. "superblocks" is the super-block tail: the 32h rows from
 * zero inside their 64h group (2^3 pages), the 64h rows from zero inside
 * their 128h group (4^3 pages), the 128h rows by the one-workgroup sweeps,
 * and back up with trilinear prolongation and sweeps against a frozen ring.
 * Coarse rows aggregate their children (uniformPressureTailAggregate) at the
 * tail's scale. Rows are in index order; a row's colour is its coordinate
 * parity, as the GPU's. */
export type TailV3 = readonly [number, number, number];
export interface TailLevel {
  readonly coord: readonly TailV3[];
  /** Six neighbour rows a row (-x,+x,-y,+y,-z,+z), -1 for none. */
  readonly near: Int32Array;
  readonly D: Float64Array;
  /** Six couplings a row, in the face order above. */
  readonly K: Float64Array;
  /** A row's parent at the next level, -1 at the top. */
  parent: Int32Array;
}
export const TAIL_FACES: readonly TailV3[] = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]];
const pack = (c: TailV3) => `${c[0]},${c[1]},${c[2]}`;
const linkOf = (coord: readonly TailV3[]): Int32Array => {
  const index = new Map(coord.map((c, i) => [pack(c), i])), near = new Int32Array(6 * coord.length).fill(-1);
  coord.forEach((c, i) => TAIL_FACES.forEach((f, j) => { near[6 * i + j] = index.get(pack([c[0] + f[0], c[1] + f[1], c[2] + f[2]])) ?? -1; }));
  return near;
};
export function tailLevel(coord: readonly TailV3[], D: Float64Array, K: Float64Array): TailLevel {
  return { coord, near: linkOf(coord), D, K, parent: new Int32Array(coord.length).fill(-1) };
}
/** The next level's rows: a row an occupied 2^3 group of rows with D > 0.
 * A child adds its D less its couplings toward the siblings that hold rows
 * (Galerkin's diagonal for piecewise-constant aggregation), and its outward
 * couplings; both scaled. */
export function uniformPressureTailAggregate(fine: TailLevel, scale: number): TailLevel {
  const index = new Map<string, number>(), coord: TailV3[] = [], D: number[] = [], K: number[] = [];
  fine.coord.forEach((c, i) => {
    if (fine.D[i]! <= 0) return;
    const g: TailV3 = [c[0] >> 1, c[1] >> 1, c[2] >> 1], key = pack(g);
    let e = index.get(key);
    if (e === undefined) { e = coord.length; index.set(key, e); coord.push(g); D.push(0); K.push(0, 0, 0, 0, 0, 0); }
    fine.parent[i] = e;
    let part = fine.D[i]!;
    for (let axis = 0; axis < 3; axis++) {
      const bit = c[axis]! & 1, inside = 2 * axis + 1 - bit, sibling = fine.near[6 * i + inside]!;
      if (sibling >= 0 && fine.D[sibling]! > 0) part -= fine.K[6 * i + inside]!;
      K[6 * e + 2 * axis + bit]! += scale * fine.K[6 * i + 2 * axis + bit]!;
    }
    D[e]! += scale * part;
  });
  return tailLevel(coord, Float64Array.from(D), Float64Array.from(K));
}
const colour = (c: TailV3) => (c[0] + c[1] + c[2]) & 1;
export function tailResidual(L: TailLevel, R: ArrayLike<number>, X: ArrayLike<number>, i: number): number {
  if (L.D[i]! <= 0) return 0;
  let r = R[i]! - L.D[i]! * X[i]!;
  for (let f = 0; f < 6; f++) { const n = L.near[6 * i + f]!; if (n >= 0 && L.D[n]! > 0) r += L.K[6 * i + f]! * X[n]!; }
  return r;
}
/** Red-black sweeps; ring(i, n): the value of neighbour n seen by row i, or
 * undefined to read the iterate. Without ring every neighbour is live. */
function sweep(L: TailLevel, R: ArrayLike<number>, X: Float64Array, sweeps: number, ring?: (i: number, n: number) => number | undefined) {
  for (let s = 0; s < sweeps; s++) for (let col = 0; col < 2; col++) for (let i = 0; i < L.coord.length; i++) {
    if (L.D[i]! <= 0 || colour(L.coord[i]!) !== col) continue;
    let off = R[i]!;
    for (let f = 0; f < 6; f++) { const n = L.near[6 * i + f]!; if (n < 0 || L.D[n]! <= 0) continue; off += L.K[6 * i + f]! * (ring?.(i, n) ?? X[n]!); }
    X[i] = off / L.D[i]!;
  }
}
/** The trilinear prolongation of parent values P to row i (0.75/0.25 an
 * axis toward the child's side, renormalised over parents with rows). */
function prolong(fine: TailLevel, coarse: TailLevel, P: Float64Array, i: number, find: (g: TailV3) => number): number {
  const c = fine.coord[i]!, G: TailV3 = [c[0] >> 1, c[1] >> 1, c[2] >> 1];
  let sum = 0, total = 0;
  for (let j = 0; j < 8; j++) {
    const d = [j & 1, (j >> 1) & 1, (j >> 2) & 1], o = c.map(v => (v & 1) ? 1 : -1);
    const m = find([G[0] + d[0]! * o[0]!, G[1] + d[1]! * o[1]!, G[2] + d[2]! * o[2]!]);
    if (m < 0 || coarse.D[m]! <= 0) continue;
    const w = (d[0] ? .25 : .75) * (d[1] ? .25 : .75) * (d[2] ? .25 : .75);
    sum += w * P[m]!; total += w;
  }
  return total > 0 ? sum / total : 0;
}
export interface TailHierarchy { readonly pages: TailLevel; readonly l6: TailLevel; readonly l7: TailLevel }
export function tailHierarchy(pages: TailLevel, scale: number): TailHierarchy {
  const l6 = uniformPressureTailAggregate(pages, scale);
  return { pages, l6, l7: uniformPressureTailAggregate(l6, scale) };
}
/** Today's tail: the 32h correction for R by sweeps from zero. */
export function tailPages(L: TailLevel, R: ArrayLike<number>, sweeps: number): Float64Array {
  const X = new Float64Array(L.coord.length); sweep(L, R, X, sweeps); return X;
}
/** The super-block tail's 32h correction for R. */
export function tailSuperblocks(h: TailHierarchy, R: ArrayLike<number>, sweeps: number, tailSweeps: number): Float64Array {
  const { pages, l6, l7 } = h;
  const finder = (L: TailLevel) => { const m = new Map(L.coord.map((c, i) => [pack(c), i])); return (g: TailV3) => m.get(pack(g)) ?? -1; };
  const find6 = finder(l6), find7 = finder(l7);
  const inGroup = (L: TailLevel, i: number, n: number) => L.parent[i]! >= 0 && L.parent[i] === L.parent[n];
  // Down: 32h from zero inside the 64h group; its residual into the 64h RHS.
  const X5 = new Float64Array(pages.coord.length);
  sweep(pages, R, X5, sweeps, (i, n) => inGroup(pages, i, n) ? undefined : 0);
  const R6 = new Float64Array(l6.coord.length);
  pages.coord.forEach((_, i) => { if (pages.parent[i]! >= 0) R6[pages.parent[i]!]! += tailResidual(pages, R, X5, i); });
  const X6 = new Float64Array(l6.coord.length);
  sweep(l6, R6, X6, sweeps, (i, n) => inGroup(l6, i, n) ? undefined : 0);
  const R7 = new Float64Array(l7.coord.length);
  l6.coord.forEach((_, i) => { if (l6.parent[i]! >= 0) R7[l6.parent[i]!]! += tailResidual(l6, R6, X6, i); });
  // The 128h rows: one workgroup's sweeps from zero.
  const X7 = new Float64Array(l7.coord.length); sweep(l7, R7, X7, tailSweeps);
  // Up: own + prolonged; the ring outside the group frozen at the same.
  const up = (F: TailLevel, RF: ArrayLike<number>, XF: Float64Array, C: TailLevel, XC: Float64Array, find: (g: TailV3) => number) => {
    const start = Float64Array.from(F.coord, (_, i) => F.D[i]! > 0 ? XF[i]! + prolong(F, C, XC, i, find) : 0), Y = Float64Array.from(start);
    sweep(F, RF, Y, sweeps, (i, n) => inGroup(F, i, n) ? undefined : start[n]!);
    return Y;
  };
  const Y6 = up(l6, R6, X6, l7, X7, find7);
  return up(pages, R, X5, l6, Y6, find6);
}
/** A box of 32h rows, one a page: open (Dirichlet) above its top layer,
 * walls elsewhere; every coupling 1. */
export function tailPool([NX, NY, NZ]: TailV3): TailLevel {
  const coord: TailV3[] = [];
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) coord.push([x, y, z]);
  const L = tailLevel(coord, new Float64Array(coord.length), new Float64Array(6 * coord.length));
  coord.forEach((c, i) => {
    let d = 0;
    TAIL_FACES.forEach((f, j) => { const k = L.near[6 * i + j]! >= 0 || (f[1] === 1 && c[1] === NY - 1) ? 1 : 0; L.K[6 * i + j] = k; d += k; });
    L.D[i] = d;
  });
  return L;
}
/** n tail corrections in turn, each for the residual the last left: the
 * 32h iterate after each (the tail's cycles within one 8h V-cycle, without
 * the 8h and 16h legs). */
export function tailIterate(L: TailLevel, R: ArrayLike<number>, tail: (residual: Float64Array) => Float64Array, n: number): Float64Array[] {
  const X = new Float64Array(L.coord.length), out: Float64Array[] = [];
  for (let k = 0; k < n; k++) {
    const r = Float64Array.from(L.coord, (_, i) => tailResidual(L, R, X, i)), y = tail(r);
    y.forEach((v, i) => { X[i]! += v; });
    out.push(Float64Array.from(X));
  }
  return out;
}
/** The exact solve (conjugate gradients to round-off). */
export function tailExact(L: TailLevel, R: ArrayLike<number>, iterations = 4000): Float64Array {
  const n = R.length, x = new Float64Array(n), r = Float64Array.from(R), p = Float64Array.from(R), Ap = new Float64Array(n);
  let rr = r.reduce((a, v) => a + v * v, 0);
  for (let it = 0; it < iterations && rr > 1e-28; it++) {
    for (let i = 0; i < n; i++) { let v = L.D[i]! * p[i]!; for (let f = 0; f < 6; f++) { const m = L.near[6 * i + f]!; if (m >= 0) v -= L.K[6 * i + f]! * p[m]!; } Ap[i] = v; }
    const a = rr / p.reduce((s, v, i) => s + v * Ap[i]!, 0);
    for (let i = 0; i < n; i++) { x[i]! += a * p[i]!; r[i]! -= a * Ap[i]!; }
    const next = r.reduce((s, v) => s + v * v, 0);
    for (let i = 0; i < n; i++) p[i] = r[i]! + next / rr * p[i]!;
    rr = next;
  }
  return x;
}
