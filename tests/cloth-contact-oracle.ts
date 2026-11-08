/**
 * Float64 CPU oracle for the OGC cloth contact (lib/cloth/contact). Test-only:
 * nothing in the step path may import it.
 *
 * It restates every rule the GPU detector and the vertex contact block apply,
 * in double precision and by brute force:
 *  - closest points (vertex-triangle with its Voronoi feature, segment-segment);
 *  - OGC block membership (Chen et al. 2025, eqs. 8-9) and Algorithm 1's set
 *    semantics, so the GPU pair lists can be compared as sets of (vertex, face);
 *  - edge-edge pairs whose closest points lie inside both edges;
 *  - the d_min terms and the conservative bound (eqs. 21-26) with per-vertex
 *    query radii;
 *  - the two-stage C2 activation (eqs. 18-20), the pair energy, its exact
 *    gradient and Hessian, and IPC lagged friction with VBD's stabilised
 *    Hessian;
 *  - exact triangle-triangle intersection between non-adjacent faces.
 */

export type V3 = [number, number, number];

export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a: V3) => Math.hypot(a[0], a[1], a[2]);

/** Feature codes shared with the WGSL: vertex k (0..2), edge k = (t[k], t[k+1]) (3..5), face (6). */
export const FEATURE_FACE = 6;

/** Ericson's closest point on a triangle, with the Voronoi feature it lies on. */
export function closestPointTriangle(p: V3, a: V3, b: V3, c: V3): { bary: V3; feature: number } {
  const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return { bary: [1, 0, 0], feature: 0 };
  const bp = sub(p, b), d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return { bary: [0, 1, 0], feature: 1 };
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); return { bary: [1 - v, v, 0], feature: 3 }; }
  const cp = sub(p, c), d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return { bary: [0, 0, 1], feature: 2 };
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); return { bary: [1 - w, 0, w], feature: 5 }; }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return { bary: [0, 1 - w, w], feature: 4 }; }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom, w = vc * denom;
  return { bary: [1 - v - w, v, w], feature: FEATURE_FACE };
}

/** Ericson's closest points between segments p0p1 and q0q1: parameters and distance. */
export function closestSegmentSegment(p0: V3, p1: V3, q0: V3, q1: V3): { s: number; t: number; d: number } {
  const d1 = sub(p1, p0), d2 = sub(q1, q0), rr = sub(p0, q0);
  const a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, rr);
  const c = dot(d1, rr), b = dot(d1, d2), denom = a * e - b * b;
  let s = denom > 0 ? Math.min(1, Math.max(0, (b * f - c * e) / denom)) : 0;
  let t = (b * s + f) / e;
  if (t < 0) { t = 0; s = Math.min(1, Math.max(0, -c / a)); }
  else if (t > 1) { t = 1; s = Math.min(1, Math.max(0, (b - c) / a)); }
  const d = norm(sub(add(p0, scale(d1, s)), add(q0, scale(d2, t))));
  return { s, t, d };
}

// ---------------------------------------------------------------- topology

export interface OracleMesh {
  x: V3[];
  tris: [number, number, number][];
  /** v0, v1, face0, face1 (NONE for a boundary edge). */
  edges: [number, number, number, number][];
}
export const NONE = 0xffffffff;

export function buildEdges(tris: [number, number, number][]): [number, number, number, number][] {
  const map = new Map<string, number>(), edges: [number, number, number, number][] = [];
  tris.forEach((t, f) => {
    for (let k = 0; k < 3; k++) {
      const a = t[k], b = t[(k + 1) % 3], key = a < b ? `${a},${b}` : `${b},${a}`;
      const e = map.get(key);
      if (e === undefined) { map.set(key, edges.length); edges.push([Math.min(a, b), Math.max(a, b), f, NONE]); }
      else edges[e][3] = f;
    }
  });
  return edges;
}

function stars(mesh: OracleMesh) {
  const vt: number[][] = mesh.x.map(() => []), ve: number[][] = mesh.x.map(() => []);
  mesh.tris.forEach((t, f) => t.forEach(v => vt[v].push(f)));
  mesh.edges.forEach((e, i) => { ve[e[0]].push(i); ve[e[1]].push(i); });
  return { vt, ve };
}

// ---------------------------------------------------------------- blocks (OGC eqs. 8-9)

/** x in the vertex block of k: (x - x_k).(x_k - x_k') >= 0 for every neighbour k'. */
export function vertexFeasible(mesh: OracleMesh, ve: number[][], x: V3, k: number): boolean {
  for (const e of ve[k]) {
    const o = mesh.edges[e][0] === k ? mesh.edges[e][1] : mesh.edges[e][0];
    if (dot(sub(x, mesh.x[k]), sub(mesh.x[k], mesh.x[o])) < 0) return false;
  }
  return true;
}

/** x in the edge block of edge e: strictly between the end planes and beyond each adjacent face. */
export function edgeFeasible(mesh: OracleMesh, x: V3, e: number): boolean {
  const [i, j, f0, f1] = mesh.edges[e], xi = mesh.x[i], xj = mesh.x[j];
  if (dot(sub(x, xi), sub(xj, xi)) <= 0 || dot(sub(x, xj), sub(xi, xj)) <= 0) return false;
  for (const f of [f0, f1]) {
    if (f === NONE) continue;
    const o = mesh.tris[f].find(v => v !== i && v !== j)!, xo = mesh.x[o];
    const dir = sub(xj, xi), s = dot(sub(xo, xi), dir) / dot(dir, dir), p = add(xi, scale(dir, s));
    if (dot(sub(x, p), sub(p, xo)) < 0) return false;
  }
  return true;
}

export function edgeIdOf(mesh: OracleMesh, a: number, b: number): number {
  const lo = Math.min(a, b), hi = Math.max(a, b);
  const e = mesh.edges.findIndex(q => q[0] === lo && q[1] === hi);
  if (e < 0) throw new Error(`no edge ${a}-${b}`);
  return e;
}

/** A global face id: "v:k", "e:id" or "f:t". */
export function globalFeature(mesh: OracleMesh, t: number, feature: number): string {
  const tri = mesh.tris[t];
  if (feature < 3) return `v:${tri[feature]}`;
  if (feature < 6) return `e:${edgeIdOf(mesh, tri[feature - 3], tri[(feature - 2) % 3])}`;
  return `f:${t}`;
}

// ---------------------------------------------------------------- brute-force detection

export interface OracleDetection {
  /** Vertex-facet pairs as "v|feature" strings. */
  vf: Set<string>;
  /** Edge-edge pairs as "e|e'" strings, both orders. */
  ee: Set<string>;
  dminV: number[];
  dminT: number[];
  dminE: number[];
  /** Conservative bounds b_v. */
  bound: number[];
  /** Pairs within `margin` of a classification threshold (excluded from strict set equality). */
  ambiguousVF: Set<string>;
  ambiguousEE: Set<string>;
}

/**
 * Algorithm 1 and 2 semantics by brute force. `rq[v]` is the vertex's query
 * radius; every d_min term is clipped at the radius of the vertex the bound is
 * for, so the result is independent of how the GPU search culls.
 */
export function bruteForceDetection(mesh: OracleMesh, r: number, rq: number[], gammaP: number, margin = 1e-6): OracleDetection {
  const { vt, ve } = stars(mesh), X = mesh.x;
  const vf = new Set<string>(), ee = new Set<string>(), ambiguousVF = new Set<string>(), ambiguousEE = new Set<string>();
  const dminV = X.map(() => Infinity), dminT = mesh.tris.map(() => Infinity), dminE = mesh.edges.map(() => Infinity);
  for (let v = 0; v < X.length; v++) {
    for (let t = 0; t < mesh.tris.length; t++) {
      const tri = mesh.tris[t];
      if (tri.includes(v)) continue;
      const a = X[tri[0]], b = X[tri[1]], c = X[tri[2]];
      const { bary, feature } = closestPointTriangle(X[v], a, b, c);
      const cp = add(add(scale(a, bary[0]), scale(b, bary[1])), scale(c, bary[2]));
      const d = norm(sub(X[v], cp));
      dminV[v] = Math.min(dminV[v], d); dminT[t] = Math.min(dminT[t], d);
      if (d >= r + margin) continue;
      const key = `${v}|${globalFeature(mesh, t, feature)}`;
      // Near a Voronoi or block boundary the f32 GPU may classify either way.
      const near = Math.abs(d - r) < margin || bary.some(q => q > 0 && q < margin) || bary.some(q => q < 1 && q > 1 - margin)
        || nearBlockBoundary(mesh, ve, X[v], t, feature, margin);
      if (d >= r) { ambiguousVF.add(key); continue; }
      let ok = true;
      if (feature < 3) ok = vertexFeasible(mesh, ve, X[v], tri[feature]);
      else if (feature < 6) ok = edgeFeasible(mesh, X[v], edgeIdOf(mesh, tri[feature - 3], tri[(feature - 2) % 3]));
      if (near) ambiguousVF.add(key);
      if (ok) vf.add(key);
    }
  }
  for (let e = 0; e < mesh.edges.length; e++) {
    const [a0, a1] = mesh.edges[e];
    for (let f = 0; f < mesh.edges.length; f++) {
      const [b0, b1] = mesh.edges[f];
      if (a0 === b0 || a0 === b1 || a1 === b0 || a1 === b1) continue;
      const lo = Math.min(e, f) === e;
      const q = lo ? closestSegmentSegment(X[a0], X[a1], X[b0], X[b1]) : swapST(closestSegmentSegment(X[b0], X[b1], X[a0], X[a1]));
      dminE[e] = Math.min(dminE[e], q.d);
      const interior = q.s > 0 && q.s < 1 && q.t > 0 && q.t < 1;
      const near = Math.abs(q.d - r) < margin || Math.min(q.s, 1 - q.s, q.t, 1 - q.t) < margin;
      if (near) ambiguousEE.add(`${e}|${f}`);
      if (q.d < r && interior) ee.add(`${e}|${f}`);
    }
  }
  const bound = X.map((_, v) => {
    let m = Math.min(rq[v], dminV[v]);
    for (const t of vt[v]) m = Math.min(m, dminT[t]);
    for (const e of ve[v]) m = Math.min(m, dminE[e]);
    return gammaP * m;
  });
  return { vf, ee, dminV, dminT, dminE, bound, ambiguousVF, ambiguousEE };
}

function swapST(q: { s: number; t: number; d: number }) { return { s: q.t, t: q.s, d: q.d }; }

function nearBlockBoundary(mesh: OracleMesh, ve: number[][], x: V3, t: number, feature: number, margin: number): boolean {
  const tri = mesh.tris[t];
  if (feature < 3) {
    const k = tri[feature];
    return ve[k].some(e => {
      const o = mesh.edges[e][0] === k ? mesh.edges[e][1] : mesh.edges[e][0];
      const dir = sub(mesh.x[k], mesh.x[o]);
      return Math.abs(dot(sub(x, mesh.x[k]), dir)) < margin * norm(dir);
    });
  }
  if (feature < 6) {
    const e = edgeIdOf(mesh, tri[feature - 3], tri[(feature - 2) % 3]);
    const [i, j, f0, f1] = mesh.edges[e], xi = mesh.x[i], xj = mesh.x[j], dir = sub(xj, xi);
    if (Math.abs(dot(sub(x, xi), dir)) < margin * norm(dir) || Math.abs(dot(sub(x, xj), dir)) < margin * norm(dir)) return true;
    for (const f of [f0, f1]) {
      if (f === NONE) continue;
      const o = mesh.tris[f].find(v => v !== i && v !== j)!, xo = mesh.x[o];
      const s = dot(sub(xo, xi), dir) / dot(dir, dir), p = add(xi, scale(dir, s)), n = sub(p, xo);
      if (Math.abs(dot(sub(x, p), n)) < margin * norm(n)) return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------- energy (OGC eqs. 18-20, IPC friction)

export interface ContactParams { r: number; k: number; mu: number; epsV: number; dt: number }

/** The two-stage activation g(d) with tau = r/2, and its first two derivatives. */
export function activation(d: number, r: number, k: number): [number, number, number] {
  if (d >= r) return [0, 0, 0];
  const tau = r / 2;
  if (d >= tau) return [0.5 * k * (r - d) ** 2, -k * (r - d), k];
  const k2 = k * tau * (r - tau), b = 0.5 * k * (r - tau) ** 2 + k2 * Math.log(tau);
  return [-k2 * Math.log(d) + b, -k2 / d, k2 / (d * d)];
}

/** IPC's f0 (energy) and f1 = f0' for the static-dynamic friction transition at eps = epsV * dt. */
export function frictionF0(y: number, eps: number) { return y >= eps ? y : -y * y * y / (3 * eps * eps) + y * y / eps + eps / 3; }
export function frictionF1OverY(y: number, eps: number) { return y >= eps ? 1 / y : 2 / eps - y / (eps * eps); }

/**
 * One contact as the vertex block evaluates it: participants and their weights,
 * so the gap is sum_i w_i x_i. Vertex-facet weights come from the projection on
 * the feature's affine hull at the current positions; edge-edge weights are the
 * frozen detection parameters (s, t).
 */
export interface OracleContact {
  kind: "vf" | "ee";
  /** vf: [v', t0, t1, t2]; ee: [e0, e1, e'0, e'1]. */
  ids: [number, number, number, number];
  feature: number;
  s: number; t: number;
  /** Lagged normal force magnitude -g'(d) at detection. */
  lambda: number;
}

export function contactWeights(c: OracleContact, X: V3[]): [number, number, number, number] {
  if (c.kind === "ee") return [1 - c.s, c.s, -(1 - c.t), -c.t];
  const [v, a, b, cc] = c.ids, p = X[v], tri = [X[a], X[b], X[cc]];
  if (c.feature < 3) { const w: [number, number, number, number] = [1, 0, 0, 0]; w[1 + c.feature] = -1; return w; }
  if (c.feature < 6) {
    const i = c.feature - 3, j = (i + 1) % 3, dir = sub(tri[j], tri[i]);
    const s = dot(sub(p, tri[i]), dir) / dot(dir, dir);
    const w: [number, number, number, number] = [1, 0, 0, 0]; w[1 + i] = -(1 - s); w[1 + j] = -s; return w;
  }
  const v0 = sub(tri[1], tri[0]), v1 = sub(tri[2], tri[0]), v2 = sub(p, tri[0]);
  const d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1), d20 = dot(v2, v0), d21 = dot(v2, v1);
  const den = d00 * d11 - d01 * d01, bv = (d11 * d20 - d01 * d21) / den, bw = (d00 * d21 - d01 * d20) / den;
  return [1, -(1 - bv - bw), -bv, -bw];
}

function gapOf(w: number[], ids: number[], X: V3[]): V3 {
  let g: V3 = [0, 0, 0];
  for (let i = 0; i < 4; i++) g = add(g, scale(X[ids[i]], w[i]));
  return g;
}

/**
 * Pair energy: normal activation of the gap distance plus lagged friction
 * mu*lambda*f0(|u|), u the tangential relative displacement since the step
 * start. `frozen` (weights, normal) evaluates the friction term with the
 * detection-time basis, as its derivative is defined.
 */
export function contactEnergy(c: OracleContact, X: V3[], Xt: V3[], p: ContactParams, frozen?: { w: number[]; n: V3 }): number {
  const w = frozen?.w ?? contactWeights(c, X), ids = c.ids;
  const g = gapOf(w, ids, X), d = norm(g), n = frozen?.n ?? scale(g, 1 / d);
  let dx: V3 = [0, 0, 0];
  for (let i = 0; i < 4; i++) dx = add(dx, scale(sub(X[ids[i]], Xt[ids[i]]), w[i]));
  const u = sub(dx, scale(n, dot(n, dx)));
  return activation(d, p.r, p.k)[0] + p.mu * c.lambda * frictionF0(norm(u), p.epsV * p.dt);
}

/** Exact normal-energy gradient on participant `role`, and its exact Hessian block. */
export function normalGradientHessian(c: OracleContact, X: V3[], p: ContactParams, role: number): { grad: V3; hess: number[][] } {
  const w = contactWeights(c, X), g = gapOf(w, c.ids, X), d = norm(g), n = scale(g, 1 / d);
  const [, g1, g2] = activation(d, p.r, p.k), wr = w[role];
  const grad = scale(n, g1 * wr);
  const hess = [0, 1, 2].map(i => [0, 1, 2].map(j => wr * wr * (g2 * n[i] * n[j] + g1 * ((i === j ? 1 : 0) - n[i] * n[j]) / d)));
  return { grad, hess };
}

/**
 * What the vertex block accumulates for participant `role`: force (minus the
 * gradient with friction's basis treated as constant) and the PSD Hessian
 * g'' w^2 n n^T + mu lambda w^2 (f1/|u|) (I - n n^T).
 */
export function vertexForceHessian(c: OracleContact, X: V3[], Xt: V3[], p: ContactParams, role: number): { f: V3; H: number[][] } {
  const w = contactWeights(c, X), ids = c.ids, g = gapOf(w, ids, X), d = norm(g), n = scale(g, 1 / d);
  const [, g1, g2] = activation(d, p.r, p.k), wr = w[role];
  let dx: V3 = [0, 0, 0];
  for (let i = 0; i < 4; i++) dx = add(dx, scale(sub(X[ids[i]], Xt[ids[i]]), w[i]));
  const u = sub(dx, scale(n, dot(n, dx))), fy = frictionF1OverY(norm(u), p.epsV * p.dt), ml = p.mu * c.lambda;
  const f = add(scale(n, -g1 * wr), scale(u, -ml * wr * fy));
  const H = [0, 1, 2].map(i => [0, 1, 2].map(j => wr * wr * (g2 * n[i] * n[j] + ml * fy * ((i === j ? 1 : 0) - n[i] * n[j]))));
  return { f, H };
}

// ---------------------------------------------------------------- triangle-triangle intersection

function orient(a: V3, b: V3, c: V3, d: V3) { return dot(sub(b, a), cross(sub(c, a), sub(d, a))); }

/** Segment pq crosses (or touches) triangle abc. Coplanar configurations report false. */
export function segmentTriangle(p: V3, q: V3, a: V3, b: V3, c: V3): boolean {
  const o1 = orient(a, b, c, p), o2 = orient(a, b, c, q);
  if (o1 === 0 && o2 === 0) return false;
  if (o1 * o2 > 0) return false;
  const s1 = orient(p, q, a, b), s2 = orient(p, q, b, c), s3 = orient(p, q, c, a);
  return (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0);
}

export function trianglesIntersect(A: V3[], B: V3[]): boolean {
  for (let k = 0; k < 3; k++) {
    if (segmentTriangle(A[k], A[(k + 1) % 3], B[0], B[1], B[2])) return true;
    if (segmentTriangle(B[k], B[(k + 1) % 3], A[0], A[1], A[2])) return true;
  }
  return false;
}

/** Brute-force count of intersecting pairs of faces that share no vertex. */
export function countIntersections(mesh: OracleMesh): number {
  let n = 0;
  for (let i = 0; i < mesh.tris.length; i++) for (let j = i + 1; j < mesh.tris.length; j++) {
    const a = mesh.tris[i], b = mesh.tris[j];
    if (a.some(v => b.includes(v))) continue;
    if (trianglesIntersect(a.map(v => mesh.x[v]), b.map(v => mesh.x[v]))) n++;
  }
  return n;
}

// ---------------------------------------------------------------- meshes for tests

/** A regular nx-by-ny sheet in the plane z = z0, alternating diagonals. */
export function sheet(nx: number, ny: number, spacing: number, origin: V3 = [0, 0, 0]): OracleMesh {
  const x: V3[] = [], tris: [number, number, number][] = [];
  for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) x.push([origin[0] + i * spacing, origin[1] + j * spacing, origin[2]]);
  const id = (i: number, j: number) => j * (nx + 1) + i;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
    if ((i + j) % 2) { tris.push([a, b, c], [a, c, d]); } else { tris.push([a, b, d], [b, c, d]); }
  }
  return { x, tris, edges: buildEdges(tris) };
}

/** Concatenate meshes into one (vertex and face ids offset). */
export function merge(...meshes: OracleMesh[]): OracleMesh {
  const x: V3[] = [], tris: [number, number, number][] = [];
  for (const m of meshes) { const o = x.length; x.push(...m.x.map(p => [...p] as V3)); tris.push(...m.tris.map(t => t.map(v => v + o) as [number, number, number])); }
  return { x, tris, edges: buildEdges(tris) };
}

/** A small deterministic generator for jitter. */
export function rng(seed: number) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// ---------------------------------------------------------------- Divide and Truncate (Chen, Hsu, Ayman, Macklin 2026)

/** A division pair: side A (1 vertex, or an edge's 2) against side B (a face's 3, or an edge's 2). */
export interface DatPair { kind: "vt" | "ee"; ids: [number, number, number, number]; nA: number }

const faceEnvelope = (rq: number[], ids: number[]) => Math.max(...ids.map(i => rq[i]));

/**
 * Every non-adjacent vertex-triangle and edge-edge pair whose distance is at most
 * the larger of the two primitives' envelopes (a face's or an edge's envelope is
 * the widest of its vertices'). Keys "v|t" and "e|f" with e < f; pairs within
 * `margin` of the threshold are listed as ambiguous.
 */
export function bruteForceDatPairs(mesh: OracleMesh, rq: number[], margin = 1e-6) {
  const X = mesh.x, vt = new Set<string>(), ee = new Set<string>(), ambiguous = new Set<string>();
  const pairs: DatPair[] = [];
  for (let v = 0; v < X.length; v++) for (let t = 0; t < mesh.tris.length; t++) {
    const tri = mesh.tris[t];
    if (tri.includes(v)) continue;
    const { bary } = closestPointTriangle(X[v], X[tri[0]], X[tri[1]], X[tri[2]]);
    const cp = add(add(scale(X[tri[0]], bary[0]), scale(X[tri[1]], bary[1])), scale(X[tri[2]], bary[2]));
    const d = norm(sub(X[v], cp)), thr = Math.max(rq[v], faceEnvelope(rq, tri));
    if (Math.abs(d - thr) < margin) ambiguous.add(`${v}|${t}`);
    if (d <= thr) { vt.add(`${v}|${t}`); pairs.push({ kind: "vt", ids: [v, tri[0], tri[1], tri[2]], nA: 1 }); }
  }
  for (let e = 0; e < mesh.edges.length; e++) for (let f = e + 1; f < mesh.edges.length; f++) {
    const [a0, a1] = mesh.edges[e], [b0, b1] = mesh.edges[f];
    if (a0 === b0 || a0 === b1 || a1 === b0 || a1 === b1) continue;
    const d = closestSegmentSegment(X[a0], X[a1], X[b0], X[b1]).d;
    const thr = Math.max(rq[a0], rq[a1], rq[b0], rq[b1]);
    if (Math.abs(d - thr) < margin) ambiguous.add(`${e}|${f}`);
    if (d <= thr) { ee.add(`${e}|${f}`); pairs.push({ kind: "ee", ids: [a0, a1, b0, b1], nA: 2 }); }
  }
  return { vt, ee, ambiguous, pairs };
}

/** Support gap of a pair along unit n: min over side A minus max over side B. */
export function supportGap(n: V3, A: V3[], B: V3[]): number {
  return Math.min(...A.map(x => dot(n, x))) - Math.max(...B.map(x => dot(n, x)));
}

const unit = (a: V3): V3 => { const l = norm(a); return l > 1e-30 ? scale(a, 1 / l) : [0, 0, 0]; };

/** The pair's separating axis at X: the closest-point axis, else the better side of the face normal or edge cross product. */
export function datSeparator(p: DatPair, X: V3[], sepEps: number): { n: V3; gap: number } | null {
  const A = p.ids.slice(0, p.nA).map(i => X[i]), B = p.ids.slice(p.nA).map(i => X[i]);
  let n0: V3, alt: V3;
  if (p.kind === "vt") {
    const { bary } = closestPointTriangle(A[0], B[0], B[1], B[2]);
    n0 = unit(sub(A[0], add(add(scale(B[0], bary[0]), scale(B[1], bary[1])), scale(B[2], bary[2]))));
    alt = unit(cross(sub(B[1], B[0]), sub(B[2], B[0])));
  } else {
    const q = closestSegmentSegment(A[0], A[1], B[0], B[1]);
    n0 = unit(sub(add(A[0], scale(sub(A[1], A[0]), q.s)), add(B[0], scale(sub(B[1], B[0]), q.t))));
    alt = unit(cross(sub(A[1], A[0]), sub(B[1], B[0])));
  }
  let best = { n: n0, gap: supportGap(n0, A, B) };
  if (best.gap >= 2 * sepEps) return best;
  for (const n of [alt, scale(alt, -1)]) { const g = supportGap(n, A, B); if (g > best.gap) best = { n, gap: g }; }
  return best.gap > 0 ? best : null;
}

/** Eq. 16 with the endpoint rule: a wrong-side start may approach its side but not retreat further. */
export function datRatio(s0: number, s1: number, gammaR: number): number {
  if (s0 < 0) return s1 >= s0 ? 1 : 0;
  if (s1 >= 0) return 1;
  return Math.min(1, Math.max(0, gammaR * s0 / (s0 - s1)));
}

/**
 * Planar-DAT (Alg. 2): the truncated positions X + t_v dX_v, each then capped at
 * cap[v]. Planes use X; each side's share of the gap is proportional to its own
 * largest approach, at least 5% (and sepEps) per side. A pair with no separator holds its vertices.
 */
export function planarTruncate(X: V3[], Xnew: V3[], pairs: DatPair[], cap: number[], gammaR: number, sepEps: number): { x: V3[]; tv: number[] } {
  const tv = X.map(() => 1);
  const dX = Xnew.map((x, i) => sub(x, X[i]));
  for (const p of pairs) {
    const sep = datSeparator(p, X, sepEps);
    if (!sep) { for (const i of p.ids) tv[i] = 0; continue; }
    const { n, gap } = sep;
    const proj = p.ids.map(i => dot(n, X[i])), dn = p.ids.map(i => dot(n, dX[i]));
    let supA = Infinity, supB = -Infinity, apA = 0, apB = 0;
    p.ids.forEach((_, k) => { if (k < p.nA) { supA = Math.min(supA, proj[k]); apA = Math.max(apA, -dn[k]); } else { supB = Math.max(supB, proj[k]); apB = Math.max(apB, dn[k]); } });
    let lam = 0.5;
    if (gap >= 2 * sepEps) {
      if (apA + apB > 0) lam = apB / (apA + apB);
      const mf = Math.max(0.05, sepEps / gap);
      lam = Math.min(1 - mf, Math.max(mf, lam));
    }
    const plane = supB + lam * gap;
    p.ids.forEach((i, k) => {
      const a = k < p.nA, s0 = (a ? proj[k] - plane : plane - proj[k]) - sepEps;
      tv[i] = Math.min(tv[i], datRatio(s0, s0 + (a ? dn[k] : -dn[k]), gammaR));
    });
  }
  const x = X.map((x0, i) => {
    let d = scale(dX[i], tv[i]);
    const l = norm(d);
    if (l > cap[i]) d = scale(d, cap[i] / l);
    return add(x0, d);
  });
  return { x, tv };
}

/** Isotropic DAT (OGC): |dX_v| <= b_v. */
export function isotropicTruncate(X: V3[], Xnew: V3[], bound: number[]): V3[] {
  return X.map((x0, i) => { let d = sub(Xnew[i], x0); const l = norm(d); if (l > bound[i]) d = scale(d, bound[i] / l); return add(x0, d); });
}

/**
 * Continuous check of the straight-line motion X0 -> X1 by brute force: some
 * non-incident vertex crosses a face or two non-adjacent edges cross. Roots of
 * the coplanarity cubic are bracketed on a fine grid and bisected in Float64.
 */
export function motionPenetrates(mesh: OracleMesh, X0: V3[], X1: V3[], samples = 64): boolean {
  const at = (i: number, s: number): V3 => add(X0[i], scale(sub(X1[i], X0[i]), s));
  const cop = (ids: number[], s: number) => { const [p, a, b, c] = ids.map(i => at(i, s)); return orient(a, b, c, p); };
  const crossesAt = (ids: number[], s: number, vt: boolean) => {
    const P = ids.map(i => at(i, s));
    if (vt) {
      const { bary } = closestPointTriangle(P[0], P[1], P[2], P[3]);
      const cp = add(add(scale(P[1], bary[0]), scale(P[2], bary[1])), scale(P[3], bary[2]));
      return norm(sub(P[0], cp)) < 1e-12;
    }
    return closestSegmentSegment(P[0], P[1], P[2], P[3]).d < 1e-12;
  };
  const test = (ids: number[], vt: boolean) => {
    // ee coplanarity: orient(p0, p1, q0, q1) shares the cubic form with vt (p, a, b, c).
    const order = vt ? ids : [ids[2], ids[0], ids[1], ids[3]];
    let s0 = 0, f0 = cop(order, 0);
    for (let k = 1; k <= samples; k++) {
      const s1 = k / samples, f1 = cop(order, s1);
      if (f0 === 0 && crossesAt(ids, s0, vt)) return true;
      if (f0 * f1 < 0) {
        let lo = s0, hi = s1, flo = f0;
        for (let j = 0; j < 60; j++) { const m = 0.5 * (lo + hi), fm = cop(order, m); if (fm * flo <= 0) hi = m; else { lo = m; flo = fm; } }
        if (crossesAt(ids, 0.5 * (lo + hi), vt)) return true;
      }
      s0 = s1; f0 = f1;
    }
    return false;
  };
  for (let v = 0; v < X0.length; v++) for (const tri of mesh.tris) if (!tri.includes(v) && test([v, ...tri], true)) return true;
  for (let e = 0; e < mesh.edges.length; e++) for (let f = e + 1; f < mesh.edges.length; f++) {
    const [a0, a1] = mesh.edges[e], [b0, b1] = mesh.edges[f];
    if (a0 === b0 || a0 === b1 || a1 === b0 || a1 === b1) continue;
    if (test([a0, a1, b0, b1], false)) return true;
  }
  return false;
}

// ------------------------------------------------------------------ static solid field (Float64)

import {
  SOLID_BRICK_CELLS, cellGradientRange, sampleClothSolidField, type ClothSolidField,
} from "../lib/cloth/contact/solid-field";

/** Exact signed distances of the test solids. */
export const boxSdf = (lo: V3, hi: V3) => {
  const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2, cz = (lo[2] + hi[2]) / 2, hx = (hi[0] - lo[0]) / 2, hy = (hi[1] - lo[1]) / 2, hz = (hi[2] - lo[2]) / 2;
  return (p: readonly number[]) => {
    const qx = Math.abs(p[0] - cx) - hx, qy = Math.abs(p[1] - cy) - hy, qz = Math.abs(p[2] - cz) - hz;
    const ox = Math.max(qx, 0), oy = Math.max(qy, 0), oz = Math.max(qz, 0);
    return Math.sqrt(ox * ox + oy * oy + oz * oz) + Math.min(Math.max(qx, qy, qz), 0);
  };
};
export const sphereSdf = (c: V3, R: number) => (p: readonly number[]) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) - R;

export interface OracleSolidBound { far: boolean; phi0: number; g0: V3; J: number; stored: boolean }

/** The GPU's ball bound (ogcSolidBallBound) in Float64 on the field's node values. */
export function solidBallBound(field: ClothSolidField, x0: V3, rho: number, r: number): OracleSolidBound {
  const s = sampleClothSolidField(field, x0);
  const out: OracleSolidBound = { far: true, phi0: s.phi, g0: [...s.grad] as V3, J: 0, stored: s.stored };
  if (Math.min(s.phi, field.band) - field.maxLipschitz * rho >= r) return out;
  const cellOf = (p: V3) => p.map((v, a) => Math.floor((v - field.origin[a]) / field.cell));
  const c0 = cellOf(sub(x0, [rho, rho, rho])), c1 = cellOf(add(x0, [rho, rho, rho]));
  let J = 0;
  for (let cz = c0[2]; cz <= c1[2]; cz++) for (let cy = c0[1]; cy <= c1[1]; cy++) for (let cx = c0[0]; cx <= c1[0]; cx++) {
    const c = [cx, cy, cz], b = c.map(v => Math.floor(v / SOLID_BRICK_CELLS));
    const brick = field.bricks.get(b.join(","));
    if (!brick) return out;
    const { lo, hi } = cellGradientRange(brick.values, c.map((v, a) => v - SOLID_BRICK_CELLS * b[a]) as V3);
    J = Math.max(J, Math.hypot(...[0, 1, 2].map(a => Math.max(Math.abs(hi[a] / field.cell - s.grad[a]), Math.abs(s.grad[a] - lo[a] / field.cell)))));
  }
  return { ...out, far: false, J };
}

/** Apply's solid factor: the displacement d from x_det scaled by 1 / max(1, j|d| - q.d). */
export function solidTruncate(b: OracleSolidBound, d: V3, gammaR: number): V3 {
  if (b.far) return d;
  if (!(b.phi0 > 0)) return [0, 0, 0];
  const s = 1 / (gammaR * b.phi0), den = b.J * s * norm(d) - s * dot(b.g0, d);
  return den > 1 ? scale(d, 1 / den) : d;
}

/** The solid entry's energy, force and Gauss-Newton Hessian at x (ogcSolidEval); the entry carries mu lambda (the solid's friction times the lagged normal force). */
export function solidContact(field: ClothSolidField, x: V3, xt: V3, lambda: number, p: ContactParams): { e: number; f: V3; H: number[][] } {
  const s = sampleClothSolidField(field, x);
  if (!(s.phi > 0)) return { e: Infinity, f: [0, 0, 0], H: [[0, 0, 0], [0, 0, 0], [0, 0, 0]] };
  const [g0, g1, g2] = activation(s.phi, p.r, p.k), g = s.grad as V3;
  let e = g0, f = scale(g, -g1);
  const H = [0, 1, 2].map(i => [0, 1, 2].map(k => g2 * g[i] * g[k]));
  if (lambda > 0) {
    const gl = norm(g), n: V3 = gl > 1e-30 ? scale(g, 1 / gl) : [0, 0, 0];
    const dx = sub(x, xt), u = sub(dx, scale(n, dot(n, dx))), un = norm(u), eps = p.epsV * p.dt;
    const ml = lambda, fy = frictionF1OverY(un, eps);
    f = sub(f, scale(u, ml * fy));
    for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) H[i][k] += ml * fy * ((i === k ? 1 : 0) - n[i] * n[k]);
    e += ml * frictionF0(un, eps);
  }
  return { e, f, H };
}
