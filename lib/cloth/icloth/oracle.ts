/**
 * Float64 references for I-Cloth's kernels (tests and benchmarks only; never a step path): the pair geometry and the
 * cubic continuous collision test, the impact-zone solve (the same augmented-Lagrangian gradient descent), and what a
 * pose is judged by: its self-intersections and its vertices inside a solid.
 */
import { ICLOTH_SOLVER } from './icloth-wgsl';
import { iclothOmega } from './icloth';

export type IcV3 = readonly [number, number, number];
const sub = (a: IcV3, b: IcV3): [number, number, number] => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: IcV3, b: IcV3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: IcV3, b: IcV3): [number, number, number] => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const axpy = (a: IcV3, s: number, b: IcV3): [number, number, number] => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
const mix = (a: IcV3, b: IcV3, t: number): [number, number, number] => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

/** Barycentric coordinates of the point of triangle (a, b, c) closest to p. */
export function icClosestTri(p: IcV3, a: IcV3, b: IcV3, c: IcV3): [number, number, number] {
  const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a), d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return [1, 0, 0];
  const bp = sub(p, b), d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return [0, 1, 0];
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); return [1 - v, v, 0]; }
  const cp = sub(p, c), d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return [0, 0, 1];
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); return [1 - w, 0, w]; }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return [0, 1 - w, w]; }
  const sum = va + vb + vc;
  if (!(sum > 0)) return [1, 0, 0];
  return [1 - vb / sum - vc / sum, vb / sum, vc / sum];
}

/** Parameters (s, t) of the closest points of segments p1 q1 and p2 q2. */
export function icClosestSeg(p1: IcV3, q1: IcV3, p2: IcV3, q2: IcV3): [number, number] {
  const d1 = sub(q1, p1), d2 = sub(q2, p2), r = sub(p1, p2), a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
  let s = 0, t = 0;
  if (a <= 1e-300 && e <= 1e-300) return [0, 0];
  if (a <= 1e-300) t = clamp01(f / e);
  else {
    const c = dot(d1, r);
    if (e <= 1e-300) s = clamp01(-c / a);
    else {
      const b = dot(d1, d2), denom = a * e - b * b;
      if (denom > 1e-14 * a * e) s = clamp01((b * f - c * e) / denom);
      t = (b * s + f) / e;
      if (t < 0) { t = 0; s = clamp01(-c / a); } else if (t > 1) { t = 1; s = clamp01((b - c) / a); }
    }
  }
  return [s, t];
}

export type IcQuad = readonly [IcV3, IcV3, IcV3, IcV3];
/** The pair's signed weights and d = sum_i w_i x_i (kind 0: vertex x0 against face x1 x2 x3; 1: edge x0 x1 against x2 x3). */
export function icFeature(kind: 0 | 1, x: IcQuad): { w: [number, number, number, number]; d: [number, number, number] } {
  if (kind === 0) {
    const b = icClosestTri(x[0], x[1], x[2], x[3]);
    return { w: [1, -b[0], -b[1], -b[2]], d: sub(x[0], axpy(axpy(axpy([0, 0, 0], b[0], x[1]), b[1], x[2]), b[2], x[3])) };
  }
  const [s, t] = icClosestSeg(x[0], x[1], x[2], x[3]);
  return { w: [s - 1, -s, 1 - t, t], d: sub(mix(x[2], x[3], t), mix(x[0], x[1], s)) };
}

export interface IcHit {
  /** 0 none; 1 an impact at a coplanarity root; 2 the pair ends within dTol; 3 an impact without a normal. */
  hit: 0 | 1 | 2 | 3;
  /** hit 1: the time of the impact in [0, 1]. */
  t?: number;
  n: [number, number, number];
  w: [number, number, number, number];
}

/** The continuous collision test of the pair moving linearly from a to b (see icCcd in icloth-wgsl.ts). */
export function icCcd(kind: 0 | 1, a: IcQuad, b: IcQuad, eta: number, dTol: number): IcHit {
  const frame = (x: IcQuad): [IcV3, IcV3, IcV3] => kind === 0 ? [sub(x[0], x[1]), sub(x[2], x[1]), sub(x[3], x[1])] : [sub(x[2], x[0]), sub(x[1], x[0]), sub(x[3], x[2])];
  const [A0, B0, C0] = frame(a), [A1, B1, C1] = frame(b), dA = sub(A1, A0), dB = sub(B1, B0), dC = sub(C1, C0);
  const N0 = cross(B0, C0), N1 = axpy(cross(dB, C0), 1, cross(B0, dC)), N2 = cross(dB, dC);
  const c = [dot(A0, N0), dot(A0, N1) + dot(dA, N0), dot(A0, N2) + dot(dA, N1), dot(dA, N2)];
  const f = (t: number) => ((c[3]! * t + c[2]!) * t + c[1]!) * t + c[0]!;
  const breaks = [0];
  const qa = 3 * c[3]!, qb = 2 * c[2]!, qc = c[1]!, stationary: number[] = [];
  if (qa !== 0) {
    const disc = qb * qb - 4 * qa * qc;
    if (disc > 0) { const q = -0.5 * (qb + (qb >= 0 ? 1 : -1) * Math.sqrt(disc)); stationary.push(q / qa); if (q !== 0) stationary.push(qc / q); }
  } else if (qb !== 0) stationary.push(-qc / qb);
  breaks.push(...stationary.filter(t => t > 0 && t < 1).sort((x, y) => x - y), 1);
  const at = (t: number): IcQuad => [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t), mix(a[3], b[3], t)];
  for (let i = 0; i + 1 < breaks.length; i++) {
    let tl = breaks[i]!, th = breaks[i + 1]!;
    const sl = Math.sign(f(tl)), sh = Math.sign(f(th));
    let t = -1, side = sl;
    if (sl === 0) t = tl;
    else if (sl * sh < 0) {
      for (let k = 0; k < 60; k++) { const tm = 0.5 * (tl + th); if (Math.sign(f(tm)) === sl) tl = tm; else th = tm; }
      t = th;
    }
    if (t < 0) continue;
    const F = icFeature(kind, at(t));
    if (dot(F.d, F.d) < eta * eta) {
      const N = axpy(N0, t, axpy(N1, t, N2)), nn = dot(N, N);
      const start = icFeature(kind, a);
      if (side === 0) side = Math.sign(dot(start.d, N));
      if (nn > 0 && side !== 0) return { hit: 1, t, n: [side * N[0] / Math.sqrt(nn), side * N[1] / Math.sqrt(nn), side * N[2] / Math.sqrt(nn)], w: F.w };
      const ds = dot(start.d, start.d);
      if (ds > 0) return { hit: 1, t, n: [start.d[0] / Math.sqrt(ds), start.d[1] / Math.sqrt(ds), start.d[2] / Math.sqrt(ds)], w: F.w };
      return { hit: 3, t, n: [0, 0, 0], w: F.w };
    }
  }
  const end = icFeature(kind, b), de = dot(end.d, end.d);
  if (de < dTol * dTol) {
    if (de > 0) return { hit: 2, n: [end.d[0] / Math.sqrt(de), end.d[1] / Math.sqrt(de), end.d[2] / Math.sqrt(de)], w: end.w };
    return { hit: 3, n: [0, 0, 0], w: end.w };
  }
  return { hit: 0, n: [0, 0, 0], w: [0, 0, 0, 0] };
}

/** An impact's constraint C(X) = n . sum_i w_i X_ids[i] - d0 >= 0 (ids < 0: unused). */
export interface IcImpact { ids: readonly [number, number, number, number]; n: IcV3; d0: number; w: readonly [number, number, number, number] }

export interface IcZoneSolveOptions {
  /** Per vertex masses and pins of the whole mesh. */
  mass: ArrayLike<number>;
  pinned: ArrayLike<number | boolean>;
  iterations: number;
  /** The step threshold (m). */
  stepThreshold: number;
  /** The violation (m) under which a solve without a step has converged. */
  feasible: number;
  mu?: number;
}

/**
 * The impact-zone solve (the paper's Algorithm 3 at ICLOTH_SOLVER's scale) in Float64: minimise
 * ICLOTH_SOLVER.objective mu / 2 sum_i (m_i / mean m) |X_i - X0_i|^2 over the impacts' vertices subject to C_j >= 0,
 * the multipliers updated when the descent stalls. Returns the positions, the iterations taken and the largest violation left.
 */
export function iclothZoneSolveOracle(x0: Float64Array, impacts: readonly IcImpact[], o: IcZoneSolveOptions): { positions: Float64Array; iterations: number; maxViolation: number; capped: boolean } {
  const N = x0.length / 3, mu = o.mu ?? ICLOTH_SOLVER.mu, zone = new Uint8Array(N);
  for (const c of impacts) for (const i of c.ids) if (i >= 0) zone[i] = 1;
  let mass = 0, free = 0;
  for (let v = 0; v < N; v++) { if (!o.pinned[v]) { mass += o.mass[v]!; free++; } }
  const q = (v: number) => 0.5 * ICLOTH_SOLVER.objective * mu * o.mass[v]! / (free ? mass / free : 1);
  const lam = new Float64Array(impacts.length);
  const C = (X: Float64Array, c: IcImpact) => {
    let s = 0;
    for (let i = 0; i < 4; i++) { const v = c.ids[i]!; if (v >= 0) s += c.w[i]! * (c.n[0] * X[3 * v]! + c.n[1] * X[3 * v + 1]! + c.n[2] * X[3 * v + 2]!); }
    return s - c.d0;
  };
  const energy = (X: Float64Array) => {
    let e = 0;
    for (let v = 0; v < N; v++) if (zone[v] && !o.pinned[v]) for (let a = 0; a < 3; a++) e += q(v) * (X[3 * v + a]! - x0[3 * v + a]!) ** 2;
    impacts.forEach((c, j) => { const ct = Math.max(lam[j]! / mu - C(X, c), 0); e += 0.5 * mu * ct * ct; });
    return e;
  };
  let prev2 = Float64Array.from(x0), prev = Float64Array.from(x0), X = Float64Array.from(x0);
  const d = new Float64Array(3 * N);
  let s = 0, k = 0, capped = false, stalled = false;
  for (; ; k++) {
    if (k > 0) {
      const w = iclothOmega(k), next = new Float64Array(3 * N);
      for (let i = 0; i < 3 * N; i++) next[i] = w * (prev[i]! + s * d[i]! - prev2[i]!) + prev2[i]!;
      prev2 = prev; prev = next; X = next;
      if (stalled) impacts.forEach((c, j) => { lam[j] = Math.max(lam[j]! - mu * C(X, c), 0); });
    }
    // The cap: the result is X^(K - 1), the last iterate with a completed line search (as the GPU's).
    if (k === o.iterations) { capped = true; X = prev2; break; }
    const g = new Float64Array(3 * N), pre = new Float64Array(N);
    for (let v = 0; v < N; v++) if (zone[v]) { pre[v] = 2 * q(v); for (let a = 0; a < 3; a++) g[3 * v + a] = 2 * q(v) * (X[3 * v + a]! - x0[3 * v + a]!); }
    impacts.forEach((c, j) => {
      const ct = Math.max(lam[j]! / mu - C(X, c), 0);
      if (ct > 0) for (let i = 0; i < 4; i++) { const v = c.ids[i]!; if (v >= 0) { for (let a = 0; a < 3; a++) g[3 * v + a]! -= mu * ct * c.w[i]! * c.n[a]!; pre[v]! += mu * c.w[i]! * c.w[i]!; } }
    });
    let slope = 0, dmax = 0;
    for (let v = 0; v < N; v++) {
      const move = zone[v] && !o.pinned[v];
      for (let a = 0; a < 3; a++) d[3 * v + a] = move ? -g[3 * v + a]! / pre[v]! : 0;
      if (move) { const l2 = d[3 * v]! ** 2 + d[3 * v + 1]! ** 2 + d[3 * v + 2]! ** 2; slope -= pre[v]! * l2; dmax = Math.max(dmax, Math.sqrt(l2)); }
    }
    const e0 = energy(X), trial = new Float64Array(3 * N);
    let found = false, violation = 0;
    for (const c of impacts) violation = Math.max(violation, -C(X, c));
    s = 0;
    for (let j = 0; j < 8 && slope < 0; j++) {
      const sj = 2 ** -j;
      for (let i = 0; i < 3 * N; i++) trial[i] = X[i]! + sj * d[i]!;
      if (energy(trial) - e0 <= ICLOTH_SOLVER.armijo * sj * slope) { s = sj; found = true; break; }
    }
    stalled = !found || s * dmax < Math.max(o.stepThreshold, ICLOTH_SOLVER.inexact * violation);
    if (stalled && violation < o.feasible) break;
  }
  let maxViolation = 0;
  for (const c of impacts) maxViolation = Math.max(maxViolation, -C(X, c));
  return { positions: X, iterations: k, maxViolation, capped };
}

/** Whether segment p q crosses triangle a b c (strictly inside both). */
function segmentHitsTriangle(p: IcV3, q: IcV3, a: IcV3, b: IcV3, c: IcV3): boolean {
  const e1 = sub(b, a), e2 = sub(c, a), dir = sub(q, p), h = cross(dir, e2), det = dot(e1, h);
  if (Math.abs(det) < 1e-300) return false;
  const s = sub(p, a), u = dot(s, h) / det;
  if (u <= 0 || u >= 1) return false;
  const qv = cross(s, e1), v = dot(dir, qv) / det;
  if (v <= 0 || u + v >= 1) return false;
  const t = dot(e2, qv) / det;
  return t > 0 && t < 1;
}

/**
 * Self-intersections of a triangle mesh pose: (edge, triangle) pairs with no shared vertex where the edge passes
 * through the triangle. A penetration-free cloth has none.
 */
export function iclothSelfIntersections(positions: ArrayLike<number>, faces: ArrayLike<number>): number {
  const T = faces.length / 3, P = (v: number): IcV3 => [positions[3 * v]!, positions[3 * v + 1]!, positions[3 * v + 2]!];
  const edges = new Map<number, [number, number]>();
  let cell = 0, N = 0;
  for (let t = 0; t < T; t++) for (let i = 0; i < 3; i++) {
    const a = faces[3 * t + i]!, b = faces[3 * t + (i + 1) % 3]!;
    N = Math.max(N, a + 1);
    const pa = P(a), pb = P(b);
    cell = Math.max(cell, Math.hypot(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]));
  }
  for (let t = 0; t < T; t++) for (let i = 0; i < 3; i++) {
    const a = faces[3 * t + i]!, b = faces[3 * t + (i + 1) % 3]!;
    edges.set(Math.min(a, b) * N + Math.max(a, b), [Math.min(a, b), Math.max(a, b)]);
  }
  if (!(cell > 0)) return 0;
  const key = (x: number, y: number, z: number) => `${x},${y},${z}`, grid = new Map<string, number[]>();
  const cells = (lo: IcV3, hi: IcV3, visit: (k: string) => void) => {
    for (let z = Math.floor(lo[2] / cell); z <= Math.floor(hi[2] / cell); z++) for (let y = Math.floor(lo[1] / cell); y <= Math.floor(hi[1] / cell); y++)
      for (let x = Math.floor(lo[0] / cell); x <= Math.floor(hi[0] / cell); x++) visit(key(x, y, z));
  };
  const box = (pts: IcV3[]): [IcV3, IcV3] => [[0, 1, 2].map(a => Math.min(...pts.map(p => p[a]!))) as unknown as IcV3, [0, 1, 2].map(a => Math.max(...pts.map(p => p[a]!))) as unknown as IcV3];
  for (let t = 0; t < T; t++) {
    const [lo, hi] = box([P(faces[3 * t]!), P(faces[3 * t + 1]!), P(faces[3 * t + 2]!)]);
    cells(lo, hi, k => { let l = grid.get(k); if (!l) grid.set(k, l = []); l.push(t); });
  }
  let count = 0;
  for (const [a, b] of edges.values()) {
    const pa = P(a), pb = P(b), [lo, hi] = box([pa, pb]), seen = new Set<number>();
    cells(lo, hi, k => {
      for (const t of grid.get(k) ?? []) {
        if (seen.has(t)) continue;
        seen.add(t);
        const i = faces[3 * t]!, j = faces[3 * t + 1]!, l = faces[3 * t + 2]!;
        if (i === a || i === b || j === a || j === b || l === a || l === b) continue;
        if (segmentHitsTriangle(pa, pb, P(i), P(j), P(l))) count++;
      }
    });
  }
  return count;
}

/** Largest relative edge strain of a pose against rest lengths (`edges`: the topology's records, v0 v1 first). */
export function iclothMaxEdgeStrain(positions: ArrayLike<number>, edges: ArrayLike<number>, rest: ArrayLike<number>): number {
  let worst = 0;
  const stride = edges.length / rest.length;
  for (let e = 0; e < rest.length; e++) {
    const a = edges[stride * e]!, b = edges[stride * e + 1]!;
    const l = Math.hypot(positions[3 * a]! - positions[3 * b]!, positions[3 * a + 1]! - positions[3 * b + 1]!, positions[3 * a + 2]! - positions[3 * b + 2]!);
    worst = Math.max(worst, l / rest[e]! - 1);
  }
  return worst;
}
