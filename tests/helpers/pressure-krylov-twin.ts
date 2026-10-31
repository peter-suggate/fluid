/** A CPU twin of the band solve's Krylov acceleration (uniform-pressure-arena.ts,
 * schedule.krylov): the V-cycle as a variable preconditioner, one cycle a
 * Krylov step. The hierarchy is the arena's: coarse rows aggregate 2^3
 * children with their links rediscretised at the coarse scale
 * (uniformPressureTailAggregate), restriction sums the children,
 * prolongation is trilinear renormalised over the parents with rows, and
 * red-black sweeps smooth. R != P^T and the links are not R A P, so the
 * cycle is a non-symmetric preconditioner.
 *
 * GCR is flexible GCR(1) (Orthomin(1) with a variable preconditioner; Vuik &
 * van der Vorst; Saad, Iterative Methods, 9.4): each step minimises the
 * residual's 2-norm over the cycle's correction, its image made orthogonal
 * to the last direction's. Its step (q.r)/(q.q) stays bounded
 * on a singular, inconsistent block (a sealed full vessel), where flexible
 * CG's (p.r)/(p.Ap) divides by the vanishing energy of the block's constant. */
import { tailLevel, tailResidual, uniformPressureTailAggregate, TAIL_FACES, type TailLevel, type TailV3 } from "./pressure-tail-twin";

/** 0 solid (Neumann), 1 liquid row, 2 air (Dirichlet zero). */
export type KrylovCell = 0 | 1 | 2;
const pack = (c: TailV3) => `${c[0]},${c[1]},${c[2]}`;
/** The liquid rows of an N box: unit links between liquid rows, a unit
 * Dirichlet face to air, nothing to solid or past the box. */
export function krylovGrid(N: TailV3, kind: (c: TailV3) => KrylovCell): TailLevel {
  const coord: TailV3[] = [];
  for (let z = 0; z < N[2]; z++) for (let y = 0; y < N[1]; y++) for (let x = 0; x < N[0]; x++) if (kind([x, y, z]) === 1) coord.push([x, y, z]);
  const L = tailLevel(coord, new Float64Array(coord.length), new Float64Array(6 * coord.length));
  coord.forEach((c, i) => {
    let d = 0;
    TAIL_FACES.forEach((f, j) => {
      const q: TailV3 = [c[0] + f[0], c[1] + f[1], c[2] + f[2]];
      const k = q.some((v, a) => v < 0 || v >= N[a]!) ? 0 : kind(q);
      if (k === 1) { L.K[6 * i + j] = 1; d += 1; } else if (k === 2) d += 1;
    });
    L.D[i] = d;
  });
  return L;
}
export function krylovHierarchy(L: TailLevel, levels: number, scale: number): TailLevel[] {
  const H = [L];
  for (let k = 1; k < levels; k++) H.push(uniformPressureTailAggregate(H[k - 1]!, scale));
  return H;
}
const colour = (c: TailV3) => (c[0] + c[1] + c[2]) & 1;
function sweep(L: TailLevel, R: ArrayLike<number>, X: Float64Array, n: number) {
  for (let s = 0; s < n; s++) for (let col = 0; col < 2; col++) for (let i = 0; i < L.coord.length; i++) {
    if (L.D[i]! <= 0 || colour(L.coord[i]!) !== col) continue;
    let off = R[i]!;
    for (let f = 0; f < 6; f++) { const m = L.near[6 * i + f]!; if (m >= 0 && L.D[m]! > 0) off += L.K[6 * i + f]! * X[m]!; }
    X[i] = off / L.D[i]!;
  }
}
function prolong(F: TailLevel, C: TailLevel, P: Float64Array, X: Float64Array) {
  const map = new Map(C.coord.map((c, i) => [pack(c), i]));
  F.coord.forEach((c, i) => {
    if (F.D[i]! <= 0) return;
    const G = c.map(v => v >> 1), o = c.map(v => (v & 1) ? 1 : -1);
    let sum = 0, total = 0;
    for (let j = 0; j < 8; j++) {
      const d = [j & 1, (j >> 1) & 1, (j >> 2) & 1];
      const m = map.get(pack([G[0]! + d[0]! * o[0]!, G[1]! + d[1]! * o[1]!, G[2]! + d[2]! * o[2]!]));
      if (m === undefined || C.D[m]! <= 0) continue;
      const w = (d[0] ? .25 : .75) * (d[1] ? .25 : .75) * (d[2] ? .25 : .75);
      sum += w * P[m]!; total += w;
    }
    if (total > 0) X[i]! += sum / total;
  });
}
/** One V-cycle on level k for R from the iterate X (in place). */
export function krylovCycle(H: TailLevel[], R: ArrayLike<number>, X: Float64Array, sweeps = 3, coarsest = 16, k = 0): void {
  const L = H[k]!;
  if (k === H.length - 1) { sweep(L, R, X, coarsest); return; }
  sweep(L, R, X, sweeps);
  const C = H[k + 1]!, RC = new Float64Array(C.coord.length);
  L.coord.forEach((_, i) => { if (L.parent[i]! >= 0) RC[L.parent[i]!]! += tailResidual(L, R, X, i); });
  const XC = new Float64Array(C.coord.length);
  krylovCycle(H, RC, XC, sweeps, coarsest, k + 1);
  prolong(L, C, XC, X);
  sweep(L, R, X, sweeps);
}
export function krylovApply(L: TailLevel, p: ArrayLike<number>): Float64Array {
  return Float64Array.from(L.coord, (_, i) => {
    if (L.D[i]! <= 0) return 0;
    let v = L.D[i]! * p[i]!;
    for (let f = 0; f < 6; f++) { const m = L.near[6 * i + f]!; if (m >= 0 && L.D[m]! > 0) v -= L.K[6 * i + f]! * p[m]!; }
    return v;
  });
}
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!; return s; };
export interface KrylovRun { readonly x: Float64Array; /** max |b - A x| before each step and after the last. */ readonly max: number[]; /** ||b - A x||_2 likewise. */ readonly l2: number[] }
function measure(L: TailLevel, b: Float64Array, x: Float64Array, run: KrylovRun) {
  const ax = krylovApply(L, x);
  let max = 0, l2 = 0;
  for (let i = 0; i < b.length; i++) { const r = b[i]! - ax[i]!; max = Math.max(max, Math.abs(r)); l2 += r * r; }
  run.max.push(max); run.l2.push(Math.sqrt(l2));
}
export type KrylovMethod = "cycles" | "gcr" | "fcg";
/** n steps of the method from x = 0: plain cycles; flexible GCR(1); or
 * flexible CG(1) (Notay 2000). The Krylov steps take the cycle's
 * correction z = B r and the operator applied to it (one apply a step; the
 * direction's A p by recurrence, A z - beta A p_prev). */
export function krylovSolve(H: TailLevel[], b: Float64Array, n: number, method: KrylovMethod): KrylovRun {
  const L = H[0]!, x = new Float64Array(b.length), run: KrylovRun = { x, max: [], l2: [] };
  measure(L, b, x, run);
  if (method === "cycles") { for (let k = 0; k < n; k++) { krylovCycle(H, b, x); measure(L, b, x, run); } return run; }
  const r = Float64Array.from(b);
  let p: Float64Array | undefined, q: Float64Array | undefined, qq = 0, pq = 0;
  for (let k = 0; k < n; k++) {
    const z = new Float64Array(b.length); krylovCycle(H, r, z);
    const az = krylovApply(L, z);
    if (!p || !q) { p = z; q = az; }
    else {
      const beta = method === "gcr" ? dot(az, q) / qq : dot(z, q) / pq;
      p = p.map((v, i) => z[i]! - beta * v) as Float64Array; q = q.map((v, i) => az[i]! - beta * v) as Float64Array;
    }
    qq = dot(q, q); pq = dot(p, q);
    const alpha = method === "gcr" ? dot(q, r) / qq : dot(p, r) / pq;
    for (let i = 0; i < x.length; i++) { x[i]! += alpha * p[i]!; r[i]! -= alpha * q[i]!; }
    measure(L, b, x, run);
  }
  return run;
}
/** Two tanks, their free surfaces at different heights, through a 2x2
 * throat in a 4 cell wall (the seep's), open above. */
export const krylovThroat: [TailV3, (c: TailV3) => KrylovCell] = [[52, 32, 16], ([x, y, z]) => {
  if (x >= 24 && x < 28) return y >= 4 && y < 6 && z >= 7 && z < 9 ? 1 : 0;
  return y < (x < 24 ? 26 : 12) ? 1 : 2;
}];
/** A seeded RHS in [-0.5, 0.5) a row. */
export function krylovRhs(n: number, seed: number): Float64Array {
  let s = seed;
  return Float64Array.from({ length: n }, () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5);
}
